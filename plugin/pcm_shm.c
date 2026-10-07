// pcm_shm.c — ALSA ioplug plugin: voice-engine SHM ring IPC (protocol: shm_voice.h)
//
// NanoPi voice link /home/i/voice/plugin/, coexists with pcm_fifo.c.
//
//   capture   (pcm.voice_mic_shm): /dev/shm/voice_mic 多读者环 —— 打开时注册读者槽
//             （CAS used 0->1，read_seq 锚定当前 write_seq），poll per-slot 唤醒 fifo
//             （引擎每周期写 token），poll_revents 排干 token + 按 avail 拷贝环→
//             ioplug 环并推进 read_seq/心跳；落后超环长则跳到 write_seq-FRAMES（追赶，
//             计 dropped）；引擎重启（epoch++）自动重注册；引擎心跳超时→POLLERR。
//   playback  (pcm.voice_spk_shm): /dev/shm/voice_spk 多写者槽 —— 打开时注册写者槽，
//             transfer 把 ioplug 环数据写入自己槽（S16），RELEASE 发布 write_seq 并向
//             写者→引擎唤醒 fifo 写 token；槽空间不足（引擎收割慢）→ withhold POLLOUT
//             形成背压；timerfd（周期=period/rate）驱动 poll_revents；drain 收尾等引擎
//             收割（有界）后注销。
//
// 唤醒语义遵循调研结论（RINGBUF_RESEARCH.md §1.2/§3.1，pcm_jack/pcm_pipewire 同款）：
// fd 只是打醒机制，数据可用性一律在 poll_revents 里按 seq 游标计算——token 丢失/
// 偷吃最多延迟一个周期，不影响正确性。

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <alsa/asoundlib.h>
#include <alsa/pcm_external.h>
#include <alsa/pcm_plugin.h>

#include "shm_voice.h"

#define ARRAY_SIZE(ary) (sizeof(ary) / sizeof((ary)[0]))

typedef struct _snd_pcm_shm_t
{
    snd_pcm_ioplug_t io;
    int is_capture;
    char shm_path[96];
    char wake_path[112];
    int shm_fd;
    void *map;
    size_t map_len;
    ino_t ino;
    uint32_t epoch;
    int slot;          /* 注册表槽下标，-1=未注册 */
    int wake_fd;       /* capture: per-slot 唤醒 fifo；playback: 写者→引擎 fifo（写端） */
    int timer_fd;      /* playback: poll_fd 驱动 */
    int rate, channels;
    snd_pcm_format_t format;
    unsigned int frame_bytes;
    volatile snd_pcm_sframes_t ptr;      /* ioplug 环内偏移（fifo 插件同款语义） */
    snd_pcm_uframes_t abs_pos;           /* playback: 已搬到槽的绝对帧数 */
    uint64_t last_probe_ms;              /* 引擎失联探测节流 */
    uint64_t dropped;                    /* 本地统计：追赶丢弃帧 */
} snd_pcm_shm_t;

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* ------------------------------------------------------------------ */
/* 映射 / 校验 / 注册表                                                */
/* ------------------------------------------------------------------ */

static int shm_map_validate(snd_pcm_shm_t *x)
{
    struct stat st;
    size_t need = x->is_capture ? sizeof(vshm_mic_t) : sizeof(vshm_spk_t);
    uint32_t magic = x->is_capture ? VSHM_MIC_MAGIC : VSHM_SPK_MAGIC;
    uint32_t rate = x->is_capture ? VSHM_MIC_RATE : VSHM_SPK_RATE;

    int fd = open(x->shm_path, O_RDWR | O_CLOEXEC);
    if (fd < 0)
    {
        SNDERR("shm: cannot open %s: %s (voice engine 未运行?)", x->shm_path, strerror(errno));
        return -errno;
    }
    if (fstat(fd, &st) < 0 || st.st_size != (off_t)need)
    {
        SNDERR("shm: %s size %lld != %zu (engine 未运行或协议版本不符)",
               x->shm_path, (long long)st.st_size, need);
        close(fd);
        return -EINVAL;
    }
    void *m = mmap(NULL, need, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED)
    {
        SNDERR("shm: mmap %s failed: %s", x->shm_path, strerror(errno));
        close(fd);
        return -errno;
    }
    uint32_t e = __atomic_load_n(x->is_capture ? &((vshm_mic_t *)m)->epoch
                                               : &((vshm_spk_t *)m)->epoch,
                                 __ATOMIC_ACQUIRE);
    uint32_t m_magic = x->is_capture ? ((vshm_mic_t *)m)->magic : ((vshm_spk_t *)m)->magic;
    uint32_t m_ver = x->is_capture ? ((vshm_mic_t *)m)->version : ((vshm_spk_t *)m)->version;
    uint32_t m_rate = x->is_capture ? ((vshm_mic_t *)m)->sample_rate
                                    : ((vshm_spk_t *)m)->sample_rate;
    if (m_magic != magic || m_ver != VSHM_PROTO_VERSION || m_rate != rate)
    {
        SNDERR("shm: %s header mismatch (magic=0x%x ver=%u rate=%u)", x->shm_path,
               m_magic, m_ver, m_rate);
        munmap(m, need);
        close(fd);
        return -EINVAL;
    }
    if (x->map)
        munmap(x->map, x->map_len);
    if (x->shm_fd >= 0)
        close(x->shm_fd);
    x->map = m;
    x->map_len = need;
    x->shm_fd = fd; /* 留着仅为保 inode 语义；不再读写 */
    x->ino = st.st_ino;
    x->epoch = e;
    return 0;
}

static void shm_slot_release(snd_pcm_shm_t *x)
{
    if (!x->map || x->slot < 0)
        return;
    if (x->is_capture)
    {
        vshm_mic_t *rb = (vshm_mic_t *)x->map;
        if (__atomic_load_n(&rb->readers[x->slot].used, __ATOMIC_RELAXED) == 1 &&
            __atomic_load_n(&rb->readers[x->slot].pid, __ATOMIC_RELAXED) == (uint32_t)getpid())
            __atomic_store_n(&rb->readers[x->slot].used, 0, __ATOMIC_RELEASE);
    }
    else
    {
        vshm_spk_t *rb = (vshm_spk_t *)x->map;
        if (__atomic_load_n(&rb->writers[x->slot].used, __ATOMIC_RELAXED) == 1 &&
            __atomic_load_n(&rb->writers[x->slot].pid, __ATOMIC_RELAXED) == (uint32_t)getpid())
            __atomic_store_n(&rb->writers[x->slot].used, 0, __ATOMIC_RELEASE);
    }
    x->slot = -1;
}

static int shm_slot_claim(snd_pcm_shm_t *x)
{
    uint32_t n = x->is_capture ? VSHM_MIC_READERS : VSHM_SPK_WRITERS;
    uint32_t pid = (uint32_t)getpid();
    for (uint32_t i = 0; i < n; i++)
    {
        uint32_t exp = 0;
        void *usedp = x->is_capture ? (void *)&((vshm_mic_t *)x->map)->readers[i].used
                                    : (void *)&((vshm_spk_t *)x->map)->writers[i].used;
        if (!__atomic_compare_exchange_n((uint32_t *)usedp, &exp, 1, 0,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            continue;
        if (x->is_capture)
        {
            vshm_mic_t *rb = (vshm_mic_t *)x->map;
            __atomic_store_n(&rb->readers[i].pid, pid, __ATOMIC_RELAXED);
            __atomic_store_n(&rb->readers[i].dropped, 0, __ATOMIC_RELAXED);
            /* 新读者从“现在”开始：不吃历史 512ms（录音语义） */
            __atomic_store_n(&rb->readers[i].read_seq,
                             __atomic_load_n(&rb->write_seq, __ATOMIC_ACQUIRE),
                             __ATOMIC_RELAXED);
            __atomic_store_n(&rb->readers[i].hb_ms, now_ms(), __ATOMIC_RELAXED);
        }
        else
        {
            vshm_spk_t *rb = (vshm_spk_t *)x->map;
            __atomic_store_n(&rb->writers[i].pid, pid, __ATOMIC_RELAXED);
            __atomic_store_n(&rb->writers[i].dropped, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&rb->writers[i].write_seq,
                             __atomic_load_n(&rb->writers[i].consumed_seq, __ATOMIC_ACQUIRE),
                             __ATOMIC_RELAXED);
            __atomic_store_n(&rb->writers[i].hb_ms, now_ms(), __ATOMIC_RELAXED);
        }
        x->slot = (int)i;
        return 0;
    }
    SNDERR("shm: registry full (%s, %u slots) — 并发客户端过多", x->shm_path, n);
    return -EBUSY;
}

/* capture: 打开本槽专属唤醒 fifo（引擎持写端，读端只收 token） */
static int shm_open_wake(snd_pcm_shm_t *x)
{
    if (x->wake_fd >= 0)
    {
        close(x->wake_fd);
        x->wake_fd = -1;
    }
    snprintf(x->wake_path, sizeof x->wake_path, "%s.wake.r%d", x->shm_path, x->slot);
    x->wake_fd = open(x->wake_path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (x->wake_fd < 0)
    {
        SNDERR("shm: open wake fifo %s failed: %s", x->wake_path, strerror(errno));
        return -errno;
    }
    return 0;
}

/* playback: 打开写者→引擎唤醒 fifo。O_RDWR 自持读写端（fifo 插件同款技巧）：
 * 引擎死亡也不会 EPIPE（否则 write 触发 SIGPIPE，且即便按线程掩蔽，write 返回
 * EPIPE 后信号处于 pending，恢复掩码的瞬间仍会被投递杀死应用——O_RDWR 根除）。 */
static int shm_open_spk_wake(snd_pcm_shm_t *x)
{
    char path[128];
    snprintf(path, sizeof path, "%s.wake", x->shm_path);
    x->wake_fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (x->wake_fd < 0)
        return -errno; /* 引擎不在；harvest 有 20ms 兜底轮询，非致命 */
    return 0;
}

static void drain_fd(int fd)
{
    char t[256];
    while (fd >= 0 && read(fd, t, sizeof t) == (ssize_t)sizeof t)
    {
    }
}

/* ------------------------------------------------------------------ */
/* 引擎失联/epoch 恢复（poll_revents 每次调用检查，代价极小）          */
/* 返回 0=正常；-EIO=引擎已死（POLLERR）                               */
/* ------------------------------------------------------------------ */
static int shm_check_engine(snd_pcm_shm_t *x)
{
    uint64_t now = now_ms();
    if (x->is_capture)
    {
        vshm_mic_t *rb = (vshm_mic_t *)x->map;
        uint32_t e = __atomic_load_n(&rb->epoch, __ATOMIC_ACQUIRE);
        if (e != x->epoch || x->slot < 0)
        { /* 引擎重启：重注册（同段复用则原地可见） */
            shm_slot_release(x);
            if (shm_map_validate(x) < 0)
                return -EIO;
            if (shm_slot_claim(x) < 0 || shm_open_wake(x) < 0)
                return -EIO;
            /* 槽号可能变化 → poll_fd 必须跟着换到新槽的唤醒 fifo（否则永远睡在旧 fifo 上） */
            x->io.poll_fd = x->wake_fd;
            SNDERR("shm: engine epoch -> %u, re-registered as reader slot %d", x->epoch, x->slot);
            return 0;
        }
        uint64_t hb = __atomic_load_n(&rb->writer_hb_ms, __ATOMIC_RELAXED);
        if (now > hb && now - hb > VSHM_HB_GRACE_MS)
        { /* 心跳陈旧：探测是否被 unlink 重建（inode 变 → 换段重注册） */
            if (now - x->last_probe_ms > 500)
            {
                x->last_probe_ms = now;
                int fd = open(x->shm_path, O_RDWR | O_CLOEXEC);
                if (fd >= 0)
                {
                    struct stat st;
                    fstat(fd, &st);
                    close(fd);
                    if (st.st_ino != x->ino)
                    {
                        shm_slot_release(x);
                        if (shm_map_validate(x) < 0 || shm_slot_claim(x) < 0 ||
                            shm_open_wake(x) < 0)
                            return -EIO;
                        x->io.poll_fd = x->wake_fd;
                        SNDERR("shm: segment replaced (epoch %u), re-registered slot %d",
                               x->epoch, x->slot);
                        return 0;
                    }
                }
            }
            if (now - hb > VSHM_HB_DEAD_MS)
                return -EIO;
        }
    }
    else
    {
        vshm_spk_t *rb = (vshm_spk_t *)x->map;
        uint32_t e = __atomic_load_n(&rb->epoch, __ATOMIC_ACQUIRE);
        if (e != x->epoch || x->slot < 0)
        {
            shm_slot_release(x);
            if (shm_map_validate(x) < 0)
                return -EIO;
            if (shm_slot_claim(x) < 0)
                return -EIO;
            if (x->wake_fd >= 0)
                shm_open_spk_wake(x);
            SNDERR("shm: engine epoch -> %u, re-registered as writer slot %d", x->epoch, x->slot);
            return 0;
        }
        uint64_t hb = __atomic_load_n(&rb->mixer_hb_ms, __ATOMIC_RELAXED);
        if (now > hb && now - hb > VSHM_HB_GRACE_MS)
        {
            if (now - x->last_probe_ms > 500)
            {
                x->last_probe_ms = now;
                int fd = open(x->shm_path, O_RDWR | O_CLOEXEC);
                if (fd >= 0)
                {
                    struct stat st;
                    fstat(fd, &st);
                    close(fd);
                    if (st.st_ino != x->ino)
                    {
                        shm_slot_release(x);
                        if (shm_map_validate(x) < 0 || shm_slot_claim(x) < 0)
                            return -EIO;
                        if (x->wake_fd >= 0)
                            shm_open_spk_wake(x);
                        SNDERR("shm: segment replaced (epoch %u), re-registered slot %d",
                               x->epoch, x->slot);
                        return 0;
                    }
                }
            }
            if (now - hb > VSHM_HB_DEAD_MS)
                return -EIO;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* capture 数据搬运：SHM 环 → ioplug 伪 mmap 环（两侧各自处理回绕）    */
/* ------------------------------------------------------------------ */
static snd_pcm_uframes_t shm_capture_pull(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    vshm_mic_t *rb = (vshm_mic_t *)x->map;
    const snd_pcm_channel_area_t *areas = snd_pcm_ioplug_mmap_areas(io);
    if (!areas || x->slot < 0)
        return 0;
    snd_pcm_uframes_t space = io->appl_ptr - io->hw_ptr + io->buffer_size;
    if (space > io->period_size)
        space = io->period_size;

    uint64_t w = __atomic_load_n(&rb->write_seq, __ATOMIC_ACQUIRE);
    uint64_t r = __atomic_load_n(&rb->readers[x->slot].read_seq, __ATOMIC_RELAXED);
    if (r > w)
    { /* 引擎 write_seq 归零（重启）残留：锚定到现在 */
        r = w;
        __atomic_store_n(&rb->readers[x->slot].read_seq, r, __ATOMIC_RELAXED);
    }
    uint64_t avail = w - r;
    if (avail > VSHM_MIC_FRAMES)
    { /* 被覆盖：跳到现存最老帧（追赶），计丢弃 */
        uint64_t drop = avail - VSHM_MIC_FRAMES;
        x->dropped += drop;
        __atomic_store_n(&rb->readers[x->slot].dropped,
                         __atomic_load_n(&rb->readers[x->slot].dropped, __ATOMIC_RELAXED) + drop,
                         __ATOMIC_RELAXED);
        r = w - VSHM_MIC_FRAMES;
        avail = VSHM_MIC_FRAMES;
    }
    snd_pcm_uframes_t frames = space < avail ? space : (snd_pcm_uframes_t)avail;
    snd_pcm_uframes_t done = 0;
    while (done < frames)
    {
        unsigned int offset = (unsigned int)x->ptr;
        unsigned int cont = io->buffer_size - offset;
        snd_pcm_uframes_t n = frames - done;
        if (n > cont)
            n = cont;
        size_t ring_pos = (size_t)(r & (uint64_t)(VSHM_MIC_FRAMES - 1));
        size_t ring_cont = VSHM_MIC_FRAMES - ring_pos;
        if (n > ring_cont)
            n = ring_cont;
        char *dst = (char *)areas->addr + (areas->first + areas->step * offset) / 8;
        memcpy(dst, &rb->data[ring_pos], n * x->frame_bytes * io->channels);
        r += n;
        done += n;
        x->ptr = (snd_pcm_sframes_t)((offset + n) % io->buffer_size);
    }
    __atomic_store_n(&rb->readers[x->slot].read_seq, r, __ATOMIC_RELAXED);
    __atomic_store_n(&rb->readers[x->slot].hb_ms, now_ms(), __ATOMIC_RELAXED);
    return done;
}

static int shm_capture_poll_revents(snd_pcm_ioplug_t *io, struct pollfd *pfds,
                                    unsigned int nfds, unsigned short *revents)
{
    (void)nfds;
    snd_pcm_shm_t *x = io->private_data;
    *revents = 0;
    if (pfds[0].revents & (POLLIN | POLLERR | POLLHUP))
        drain_fd(x->wake_fd); /* token 只是提示：非阻塞排干 */
    if (shm_check_engine(x) < 0)
    {
        *revents = POLLERR; /* 引擎真死：明确报错而非无声挂死 */
        return 0;
    }
    if (pfds[0].revents & POLLHUP)
    { /* 引擎写端消失但仍在宽限期内（可能在重启）：小睡等待 epoch 恢复 */
        usleep(100000);
        return 0;
    }
    snd_pcm_uframes_t got = shm_capture_pull(io);
    if (got > 0)
        *revents = POLLIN;
    return 0;
}

/* ------------------------------------------------------------------ */
/* playback 数据搬运：ioplug 伪 mmap 环 → 本写者槽（背压：空间不足即止）*/
/* ------------------------------------------------------------------ */
static snd_pcm_uframes_t shm_playback_push(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    vshm_spk_t *rb = (vshm_spk_t *)x->map;
    const snd_pcm_channel_area_t *areas = snd_pcm_ioplug_mmap_areas(io);
    if (!areas || x->slot < 0)
        return 0;
    snd_pcm_uframes_t pending = io->appl_ptr - x->abs_pos;
    if (pending > io->period_size)
        pending = io->period_size;
    if (pending == 0)
        return 0;

    uint64_t w = __atomic_load_n(&rb->writers[x->slot].write_seq, __ATOMIC_RELAXED);
    uint64_t c = __atomic_load_n(&rb->writers[x->slot].consumed_seq, __ATOMIC_ACQUIRE);
    if (c > w)
    { /* 引擎重启槽被清：write_seq 锚定到 consumed */
        w = c;
        __atomic_store_n(&rb->writers[x->slot].write_seq, w, __ATOMIC_RELAXED);
    }
    uint64_t space = c + VSHM_SPK_FRAMES - w; /* 引擎未收割的占位 */
    snd_pcm_uframes_t frames = pending < space ? pending : (snd_pcm_uframes_t)space;
    snd_pcm_uframes_t done = 0;
    while (done < frames)
    {
        unsigned int offset = (unsigned int)(x->abs_pos % io->buffer_size);
        unsigned int cont = io->buffer_size - offset;
        snd_pcm_uframes_t n = frames - done;
        if (n > cont)
            n = cont;
        size_t ring_pos = (size_t)(w & (uint64_t)(VSHM_SPK_FRAMES - 1));
        size_t ring_cont = VSHM_SPK_FRAMES - ring_pos;
        if (n > ring_cont)
            n = ring_cont;
        const char *src = (const char *)areas->addr + (areas->first + areas->step * offset) / 8;
        memcpy(&rb->data[x->slot][ring_pos], src, n * x->frame_bytes * io->channels);
        w += n;
        done += n;
        x->abs_pos += n;
        x->ptr = (snd_pcm_sframes_t)((x->ptr + n) % io->buffer_size);
    }
    if (done > 0)
    {
        __atomic_store_n(&rb->writers[x->slot].write_seq, w, __ATOMIC_RELEASE);
        __atomic_store_n(&rb->writers[x->slot].hb_ms, now_ms(), __ATOMIC_RELAXED);
        char tok = 1; /* 写者→引擎 token；失败无所谓，引擎有 20ms 兜底轮询 */
        if (x->wake_fd >= 0)
        {
            /* 双保险：按线程掩蔽 SIGPIPE，且 write 得 EPIPE 后先吃掉 pending 的
             * 信号再恢复掩码（否则恢复瞬间投递）。O_RDWR 下正常不会走到这里。 */
            sigset_t block, old;
            sigemptyset(&block);
            sigaddset(&block, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &block, &old);
            ssize_t wr = write(x->wake_fd, &tok, 1);
            if (wr < 0 && errno == EPIPE)
            {
                struct timespec zt = {0, 0};
                while (sigtimedwait(&block, NULL, &zt) == -1 && errno == EINTR)
                {
                }
            }
            pthread_sigmask(SIG_SETMASK, &old, NULL);
            (void)wr;
        }
    }
    return done;
}

static int shm_playback_poll_revents(snd_pcm_ioplug_t *io, struct pollfd *pfds,
                                     unsigned int nfds, unsigned short *revents)
{
    (void)nfds;
    snd_pcm_shm_t *x = io->private_data;
    *revents = 0;
    if (pfds[0].revents & POLLIN)
        drain_fd(x->timer_fd); /* timerfd 计数排干 */
    if (shm_check_engine(x) < 0)
    {
        *revents = POLLERR;
        return 0;
    }
    shm_playback_push(io);
    vshm_spk_t *rb = (vshm_spk_t *)x->map;
    if (x->slot >= 0)
    {
        uint64_t w = __atomic_load_n(&rb->writers[x->slot].write_seq, __ATOMIC_RELAXED);
        uint64_t c = __atomic_load_n(&rb->writers[x->slot].consumed_seq, __ATOMIC_ACQUIRE);
        uint64_t space = c + VSHM_SPK_FRAMES - w;
        __atomic_store_n(&rb->writers[x->slot].hb_ms, now_ms(), __ATOMIC_RELAXED);
        if (space >= 1)
            *revents = POLLOUT; /* 有可推进空间即放行（部分写合法，aplay 自行续写） */
    }
    return 0;
}

/* playback drain：把 ioplug 环剩余数据推入槽，并等引擎收割完（有界 3s） */
#define SHM_DRAIN_POLL_MS 20
#define SHM_DRAIN_TRIES 150

static int shm_playback_drain(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    int tries = 0;
    while (tries < SHM_DRAIN_TRIES)
    {
        if (shm_check_engine(x) < 0)
            break;
        shm_playback_push(io);
        vshm_spk_t *rb = (vshm_spk_t *)x->map;
        if (x->slot >= 0)
        {
            uint64_t w = __atomic_load_n(&rb->writers[x->slot].write_seq, __ATOMIC_RELAXED);
            uint64_t c = __atomic_load_n(&rb->writers[x->slot].consumed_seq, __ATOMIC_ACQUIRE);
            int flush_left = (x->abs_pos < io->appl_ptr);
            if (!flush_left && c >= w)
                return 0; /* 全部收割完 */
        }
        usleep(SHM_DRAIN_POLL_MS * 1000);
        tries++;
    }
    return 0; /* 有界放弃（引擎不在/极慢）：应用侧无感退出 */
}

/* ------------------------------------------------------------------ */
/* 生命周期回调                                                        */
/* ------------------------------------------------------------------ */
static int shm_start(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    x->ptr = 0;
    if (!x->is_capture)
    {
        x->abs_pos = 0;
        /* timerfd 驱动 playback poll_revents（周期 = period/rate，下限 5ms） */
        unsigned int period_ns = 5000000u;
        if (io->rate > 0 && io->period_size > 0)
            period_ns = (unsigned int)((double)io->period_size / io->rate * 1e9);
        if (period_ns < 5000000u)
            period_ns = 5000000u;
        struct itimerspec its = {};
        its.it_interval.tv_nsec = period_ns;
        its.it_value.tv_nsec = period_ns;
        timerfd_settime(x->timer_fd, 0, &its, NULL);
        return 0;
    }
    /* capture：锚定到当前 write_seq（不吃 start 前积压），心跳刷新 */
    vshm_mic_t *rb = (vshm_mic_t *)x->map;
    if (x->slot >= 0)
    {
        __atomic_store_n(&rb->readers[x->slot].read_seq,
                         __atomic_load_n(&rb->write_seq, __ATOMIC_ACQUIRE),
                         __ATOMIC_RELAXED);
        __atomic_store_n(&rb->readers[x->slot].hb_ms, now_ms(), __ATOMIC_RELAXED);
    }
    return 0;
}

static int shm_stop(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    if (!x->is_capture && x->timer_fd >= 0)
    {
        struct itimerspec its = {};
        timerfd_settime(x->timer_fd, 0, &its, NULL);
    }
    return 0;
}

/* 资源回收（close 回调与 open 失败路径共用；全部置空/置 -1 保证可重入）。
 * 注意：不 free(x) —— 与 fifo 插件一致，struct 由 open 的 fail 路径释放
 * （alsa-lib snd_pcm_ioplug_close 只 free 它自己的 ioplug_priv_t）。 */
static void shm_cleanup_resources(snd_pcm_shm_t *x)
{
    shm_slot_release(x);
    if (x->wake_fd >= 0)
    {
        close(x->wake_fd);
        x->wake_fd = -1;
    }
    if (x->timer_fd >= 0)
    {
        close(x->timer_fd);
        x->timer_fd = -1;
    }
    if (x->map)
    {
        munmap(x->map, x->map_len);
        x->map = NULL;
        x->map_len = 0;
    }
    if (x->shm_fd >= 0)
    {
        close(x->shm_fd);
        x->shm_fd = -1;
    }
}

static int shm_close(snd_pcm_ioplug_t *io)
{
    shm_cleanup_resources(io->private_data);
    return 0;
}

static snd_pcm_sframes_t shm_pointer(snd_pcm_ioplug_t *io)
{
    snd_pcm_shm_t *x = io->private_data;
    return x->ptr;
}

static snd_pcm_ioplug_callback_t shm_capture_callback = {
    .start = shm_start,
    .stop = shm_stop,
    .close = shm_close,
    .pointer = shm_pointer,
    .poll_revents = shm_capture_poll_revents,
};

static snd_pcm_ioplug_callback_t shm_playback_callback = {
    .start = shm_start,
    .stop = shm_stop,
    .close = shm_close,
    .pointer = shm_pointer,
    .poll_revents = shm_playback_poll_revents,
    .drain = shm_playback_drain,
};

static int shm_hw_constraint(snd_pcm_shm_t *x)
{
    unsigned int accesses[] = {
        SND_PCM_ACCESS_RW_INTERLEAVED,
        SND_PCM_ACCESS_MMAP_INTERLEAVED};
    unsigned int formats[] = {x->format};
    int err;

    if ((err = snd_pcm_ioplug_set_param_list(&x->io, SND_PCM_IOPLUG_HW_ACCESS,
                                             ARRAY_SIZE(accesses), accesses)) < 0 ||
        (err = snd_pcm_ioplug_set_param_list(&x->io, SND_PCM_IOPLUG_HW_FORMAT,
                                             ARRAY_SIZE(formats), formats)) < 0 ||
        (err = snd_pcm_ioplug_set_param_minmax(&x->io, SND_PCM_IOPLUG_HW_CHANNELS,
                                               x->channels, x->channels)) < 0 ||
        (err = snd_pcm_ioplug_set_param_minmax(&x->io, SND_PCM_IOPLUG_HW_RATE,
                                               x->rate, x->rate)) < 0)
    {
        SNDERR("shm: ioplug cannot set params!");
        return err;
    }
    err = snd_pcm_ioplug_set_param_minmax(&x->io, SND_PCM_IOPLUG_HW_BUFFER_BYTES,
                                          256, 4 * 1024 * 1024);
    if (err < 0)
    {
        SNDERR("shm: ioplug cannot set hw buffer bytes");
        return err;
    }
    /* period 上限 = SHM 侧缓冲容量（mic 环 8192 帧 / spk 槽 4096 帧）：
       period 大于设备缓冲会让“空间>=period”的 POLLOUT 永远无法满足（实测死锁）。 */
    unsigned int period_max = x->is_capture ? 2u * VSHM_MIC_FRAMES : 768u; /* spk 钉 8ms(768B=384帧)：token 节拍=全链 8ms 对齐，防 ~21ms 大 period 超出引擎滞回底线(12ms) 唤醒容忍 → pb_err 144/24s */
    err = snd_pcm_ioplug_set_param_minmax(&x->io, SND_PCM_IOPLUG_HW_PERIOD_BYTES,
                                          128, period_max);
    if (err < 0)
    {
        SNDERR("shm: ioplug cannot set hw period bytes");
        return err;
    }
    err = snd_pcm_ioplug_set_param_minmax(&x->io, SND_PCM_IOPLUG_HW_PERIODS, 3, 1024);
    if (err < 0)
    {
        SNDERR("shm: ioplug cannot set hw periods");
        return err;
    }
    return 0;
}

/*
 * Main entry point:  pcm.voice_mic_shm { type vshm shm "/dev/shm/voice_mic" ... }
 * （type 名用 vshm：alsa-lib 自带内置 "shm" 客户端/服务器插件，同名会被内置版截胡）
 */
SND_PCM_PLUGIN_DEFINE_FUNC(vshm)
{
    snd_config_iterator_t i, next;
    const char *shm_path = NULL;
    long rate_cfg = 0, channels_cfg = 0;
    const char *fmt_cfg = NULL;
    snd_pcm_shm_t *x;
    int err;

    snd_config_for_each(i, next, conf)
    {
        snd_config_t *n = snd_config_iterator_entry(i);
        const char *id;
        if (snd_config_get_id(n, &id) < 0)
            continue;
        if (strcmp(id, "shm") == 0)
        {
            if (snd_config_get_string(n, &shm_path) < 0)
            {
                SNDERR("shm: invalid type for %s", id);
                return -EINVAL;
            }
            continue;
        }
        if (strcmp(id, "rate") == 0)
        {
            if (snd_config_get_integer(n, &rate_cfg) < 0)
            {
                SNDERR("shm: invalid type for %s", id);
                return -EINVAL;
            }
            continue;
        }
        if (strcmp(id, "format") == 0)
        {
            if (snd_config_get_string(n, &fmt_cfg) < 0)
            {
                SNDERR("shm: invalid type for %s", id);
                return -EINVAL;
            }
            continue;
        }
        if (strcmp(id, "channels") == 0)
        {
            if (snd_config_get_integer(n, &channels_cfg) < 0)
            {
                SNDERR("shm: invalid type for %s", id);
                return -EINVAL;
            }
            continue;
        }
    }
    if (!shm_path)
    {
        SNDERR("shm: 'shm' path is not set (e.g. \"/dev/shm/voice_mic\")");
        return -EINVAL;
    }

    x = calloc(1, sizeof(*x));
    if (!x)
    {
        SNDERR("shm: cannot allocate");
        return -ENOMEM;
    }
    x->is_capture = (stream == SND_PCM_STREAM_CAPTURE);
    x->shm_fd = -1;
    x->wake_fd = -1;
    x->timer_fd = -1;
    x->slot = -1;
    snprintf(x->shm_path, sizeof x->shm_path, "%s", shm_path);
    x->format = fmt_cfg ? snd_pcm_format_value(fmt_cfg) : SND_PCM_FORMAT_S16_LE;
    if (x->format != SND_PCM_FORMAT_S16_LE)
    {
        SNDERR("shm: only S16_LE supported (got %s)", fmt_cfg ? fmt_cfg : "?");
        free(x);
        return -EINVAL;
    }

    err = shm_map_validate(x);
    if (err < 0)
        goto fail;
    /* header 是格式的权威来源；asound.conf 显式配置若不符则报错（防错接线） */
    x->rate = x->is_capture ? ((vshm_mic_t *)x->map)->sample_rate
                            : ((vshm_spk_t *)x->map)->sample_rate;
    x->channels = x->is_capture ? ((vshm_mic_t *)x->map)->channels
                                : ((vshm_spk_t *)x->map)->channels;
    if ((rate_cfg && rate_cfg != x->rate) || (channels_cfg && channels_cfg != x->channels))
    {
        SNDERR("shm: asound.conf rate/channels (%ld/%ld) != segment header (%d/%d)",
               rate_cfg, channels_cfg, x->rate, x->channels);
        err = -EINVAL;
        goto fail;
    }
    x->frame_bytes = snd_pcm_format_width(x->format) / 8;

    err = shm_slot_claim(x);
    if (err < 0)
        goto fail;
    if (x->is_capture)
    {
        err = shm_open_wake(x);
        if (err < 0)
            goto fail;
    }
    else
    {
        shm_open_spk_wake(x); /* 失败非致命（引擎 20ms 兜底轮询） */
        x->timer_fd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
        if (x->timer_fd < 0)
        {
            SNDERR("shm: timerfd_create failed: %s", strerror(errno));
            err = -errno;
            goto fail;
        }
    }

    x->io.version = SND_PCM_IOPLUG_VERSION;
    x->io.name = "ALSA <-> voice-engine SHM ring Plugin";
    x->io.mmap_rw = 1;
    x->io.poll_fd = x->is_capture ? x->wake_fd : x->timer_fd;
    x->io.poll_events = POLLIN;
    x->io.callback = x->is_capture ? &shm_capture_callback : &shm_playback_callback;
    x->io.private_data = x;

    err = snd_pcm_ioplug_create(&x->io, name, stream, mode);
    if (err < 0)
        goto fail;
    if ((err = shm_hw_constraint(x)) < 0)
    {
        /* ioplug_delete 会走 close 回调回收资源；之后仅剩 free */
        snd_pcm_ioplug_delete(&x->io);
        free(x);
        return err;
    }
    *pcmp = x->io.pcm;
    return 0;

fail:
    shm_cleanup_resources(x);
    free(x);
    return err;
}

SND_PCM_PLUGIN_SYMBOL(vshm);
