/* ring_proto_test.c — SHM ring 协议草案极小验证（非生产代码）
 *
 * 验证 §5 草案的四个核心语义（无 eventfd 的 macOS 上用 pipe 等价模拟，
 * Linux 上 pipe 与 eventfd 的 poll 语义相同；pcm_jack.c 用 socketpair 同构）：
 *   1. SPSC/MRM 数据环：2^n 帧，read_seq/write_seq 单调 u64，pos = seq & mask
 *   2. 内存序：writer 数据 store-release write_seq；reader load-acquire write_seq 再读数据
 *   3. 写者覆盖策略：writer 永不阻塞，最慢读者落后超环长即被覆盖
 *   4. 读者追赶：读者发现 read_seq < write_seq - frames 时跳到新数据（丢帧计数），不回读旧洞
 *   5. 多读者互不分流：两个读者各自独立游标，同一份数据都能读到（fifo 做不到的点）
 *   6. 唤醒只是提示：poll(pipe) 醒来后必须以 write_seq>read_seq 判定数据可用（多读者共享一个唤醒 fd，
 *      事件会被任一读者消费，其余读者靠 avail 判定 —— pcm_jack/pcm_pipewire 的 poll_revents 模式）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <pthread.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>

#define N_READERS 2
#define LOG2_FRAMES 13                      /* 8192 帧 */
#define FRAMES ((unsigned)1 << LOG2_FRAMES)
#define MASK (FRAMES - 1)

typedef struct {
    int32_t used;                           /* slot 占用（CAS 分配） */
    int32_t pid;
    uint64_t read_seq;                      /* 该读者已消费的总帧数 */
    uint64_t heartbeat;                     /* 毫秒心跳，僵尸回收用 */
    uint64_t dropped;                       /* 覆盖追赶累计丢帧（诊断） */
} reader_slot_t;

typedef struct {
    uint32_t magic;                         /* 'V','R','B','1' */
    uint32_t frame_bits;                    /* 16 (S16 mono) */
    uint32_t log2_frames;
    uint32_t n_slots;
    uint64_t write_seq;                     /* 总写入帧数（单调） */
    uint64_t writer_hb;
    reader_slot_t readers[N_READERS];
    /* 数据环紧随 header（同一 SHM 段） */
    int16_t data[FRAMES];
} ring_t;

static ring_t *R;
static int wake_fd[2];                      /* writer 写 wake_fd[1]，读者 poll wake_fd[0] */
static volatile int done;
static uint64_t now_ms(void)
{
    struct timespec ts = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000ull + ts.tv_nsec / 1000000;
}

/* ---- writer 侧（对应 voice-engine BF 输出点） ---- */
static void writer_push(int16_t *src, uint32_t n)
{
    uint64_t w = __atomic_load_n(&R->write_seq, __ATOMIC_RELAXED);
    for (uint32_t i = 0; i < n; i++) {
        R->data[(w + i) & MASK] = src[i];   /* 1. 先写数据 */
    }
    __atomic_store_n(&R->write_seq, w + n, __ATOMIC_RELEASE); /* 2. 后发布 seq */
    R->writer_hb = now_ms();
    char c = 1;
    if (write(wake_fd[1], &c, 1)) { /* 唤醒只是提示 */ }
}

static void *writer_thread(void *arg)
{
    (void)arg;
    uint64_t f = 0;                          /* 帧计数器即数据 pattern */
    while (!done) {
        int16_t blk[16];
        for (int i = 0; i < 16; i++) blk[i] = (int16_t)(f++ & 0xffff);
        writer_push(blk, 16);                /* 16 帧/毫秒 = 16k 帧/秒 */
        usleep(1000);
    }
    return NULL;
}

/* ---- reader 侧（对应 ioplug poll_revents + transfer） ---- */
typedef struct { int idx; int slow; uint64_t got, dropped, polls, wakeups, resync; } rarg_t;

/* 读者消费一步：返回拿到的帧数；模拟 ioplug 从 ring 拷贝到 mmap areas。
 * 返回 1 于 *resynced 表示发生了覆盖追赶（消费方用于重新锚定 pattern 校验）。 */
static uint32_t reader_pull(int idx, int16_t *dst, uint32_t want, int *resynced)
{
    reader_slot_t *s = &R->readers[idx];
    uint64_t r = __atomic_load_n(&s->read_seq, __ATOMIC_RELAXED);
    uint64_t w = __atomic_load_n(&R->write_seq, __ATOMIC_ACQUIRE); /* 与写者 release 配对 */
    uint64_t avail = w - r;
    *resynced = 0;
    if (avail == 0) return 0;
    if (avail > FRAMES) {                    /* 4. 已被覆盖：追赶，跳到现存最老帧 */
        uint64_t gap = avail - FRAMES;
        s->dropped += gap;
        r += gap;                            /* 不回读已覆盖的洞 */
        avail = FRAMES;
        *resynced = 1;
    }
    uint32_t n = want < avail ? want : (uint32_t)avail;
    for (uint32_t i = 0; i < n; i++)
        dst[i] = R->data[(r + i) & MASK];
    __atomic_store_n(&s->read_seq, r + n, __ATOMIC_RELAXED);
    s->heartbeat = now_ms();
    return n;
}

static void *reader_thread(void *arg)
{
    rarg_t *a = arg;
    reader_slot_t *s = &R->readers[a->idx];
    __atomic_store_n(&s->used, 1, __ATOMIC_RELAXED);
    int valid = 0;                           /* pattern 校验是否已锚定 */
    uint64_t expect = 0;                     /* 下一帧的帧号 */
    while (!done) {
        struct pollfd pfd = { .fd = wake_fd[0], .events = POLLIN };
        int pr = poll(&pfd, 1, 50);
        a->polls++;
        if (pr > 0 && (pfd.revents & POLLIN)) {
            char buf[64];
            a->wakeups++;
            while (read(wake_fd[0], buf, sizeof buf) > 0) { }   /* 读干，EAGAIN 退出 */
        }
        /* 6. 醒来后不看 pipe 状态，只看自己的游标 —— 共享唤醒 fd 的关键 */
        for (;;) {
            int16_t blk[256];
            int resynced = 0;
            uint32_t n = reader_pull(a->idx, blk, a->slow ? 256 : 64, &resynced);
            if (n == 0) break;
            if (resynced) { a->resync++; valid = 0; }  /* 覆盖追赶后重新锚定 */
            if (valid) {
                for (uint32_t i = 0; i < n; i++) {
                    if ((uint64_t)(uint16_t)blk[i] != (expect & 0xffff)) {
                        printf("FAIL: reader%d discontinuity at +%u (expect %" PRIu64 ")\n",
                               a->idx, i, expect);
                        exit(1);
                    }
                    expect++;
                }
            } else {
                expect = ((uint64_t)(uint16_t)blk[n - 1]) + 1;  /* 锚定到本块末尾 */
                valid = 1;
            }
            a->got += n;
        }
        if (a->slow) usleep(700000);        /* 慢读者：700ms 消费一次，16k/s 下
                                              * 700ms 产 ~11200 帧 > 环容量 8192 → 必然覆盖追赶 */
    }
    return NULL;
}

int main(void)
{
    R = mmap(NULL, sizeof(ring_t), PROT_READ | PROT_WRITE,
             MAP_ANON | MAP_SHARED, -1, 0);
    assert(R);
    memset(R, 0, sizeof *R);
    R->magic = 0x31425256; /* "VRB1" */
    R->frame_bits = 16; R->log2_frames = LOG2_FRAMES; R->n_slots = N_READERS;
    assert(!pipe(wake_fd));
    for (int i = 0; i < 2; i++) {           /* 非阻塞：排水循环靠 EAGAIN 终止 */
        int fl = fcntl(wake_fd[i], F_GETFL);
        fcntl(wake_fd[i], F_SETFL, fl | O_NONBLOCK);
    }

    pthread_t wt, rt[N_READERS];
    rarg_t ra[N_READERS] = {{.idx = 0, .slow = 0}, {.idx = 1, .slow = 1}};
    pthread_create(&wt, NULL, writer_thread, NULL);
    for (int i = 0; i < N_READERS; i++) pthread_create(&rt[i], NULL, reader_thread, &ra[i]);

    sleep(3);
    done = 1;
    char c = 1; if (write(wake_fd[1], &c, 1)) { }
    pthread_join(wt, NULL);
    for (int i = 0; i < N_READERS; i++) pthread_join(rt[i], NULL);

    uint64_t wtotal = R->write_seq;
    printf("writer total      : %" PRIu64 " frames (expect ~48000)\n", wtotal);
    printf("reader0 (fast)    : got %" PRIu64 " (%.1f%%), polls %" PRIu64 ", wakeups %" PRIu64 ", resync %" PRIu64 "\n",
           ra[0].got, 100.0 * ra[0].got / wtotal, ra[0].polls, ra[0].wakeups, ra[0].resync);
    printf("reader1 (slow)    : got %" PRIu64 " (%.1f%%), polls %" PRIu64 ", wakeups %" PRIu64 ", resync %" PRIu64 ", dropped %" PRIu64 "\n",
           ra[1].got, 100.0 * ra[1].got / wtotal, ra[1].polls, ra[1].wakeups, ra[1].resync,
           R->readers[1].dropped);
    int ok = (ra[0].got == wtotal) && (ra[1].got < ra[0].got);
    printf("%s: fast reader lossless=%s ; slow reader consumed less than fast (no split!)=%s\n",
           ok ? "PASS" : "FAIL", ra[0].got == wtotal ? "yes" : "NO",
           ra[1].got < ra[0].got ? "yes" : "NO");
    return ok ? 0 : 1;
}
