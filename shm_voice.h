// shm_voice.h — SHM IPC 协议公共头（voice_engine 写端/读端 与 ALSA ioplug 插件 pcm_shm.c 共用）
//
// 两个独立共享内存段（plain tmpfs 文件，非 shm_open 命名空间——同一回事，路径即可发现）：
//
//   /dev/shm/voice_mic  引擎=单写者，应用=多读者（≤4）
//     header{magic,epoch,sr=16000,S16,mono,ring=8192(2^13),write_seq,writer_hb}
//       + 读者注册表[4]{used,pid,read_seq,hb_ms,dropped} + S16 数据环
//     语义：读者注册→按自身 read_seq 拷贝→更新游标+心跳；落后超环长则跳到
//     write_seq-FRAMES（追赶，计入 dropped）；写者覆盖旧数据、永不等待。
//
//   /dev/shm/voice_spk  应用=多写者（≤8），引擎=单读者/混音者
//     header{magic,epoch,sr=48000,...,mixer_hb,mixed_frames}
//       + 写者注册表[8]{used,pid,write_seq,consumed_seq,hb_ms,dropped}
//       + 每写者独立 S16 环 data[8][4096]
//     语义：写者各写各槽并推进 write_seq；引擎每周期收割各槽新增数据 float 求和
//     混音+限幅 → speaker/ref 路径，并推进 consumed_seq（写者据此算空间做背压）；
//     槽内积压超环长 → 引擎追赶（跳到最新，计入 dropped）。
//
// 唤醒（数据与唤醒分离，唤醒只是提示、数据可用性一律以 seq 游标为准）：
//   mic:  每个读者槽一条 tmpfs fifo  /dev/shm/voice_mic.wake.r<slot>，
//         引擎持 O_RDWR|O_NONBLOCK，每次发布数据后向活跃槽各写 1 字节 token
//         （per-reader 通道，天然无 token 偷吃；读端非阻塞排干）。
//   spk:  一条 tmpfs fifo /dev/shm/voice_spk.wake（写者→引擎方向），写者每次
//         写槽后写 1 字节 token，引擎读侧 poll 它（20ms 超时兜底轮询）。
//         应用侧播放插件用 timerfd（周期 = period/rate）驱动 poll_revents。
//
// 原子性：所有跨进程字段用 GCC __atomic 内建访问（armv7 cortex-a7 上 32/64 位
// 均生成 ldrex(d)/strex(d)）；write_seq/consumed_seq 发布用 RELEASE、读取用
// ACQUIRE，与数据写入配对。header 其余字段单写者原则（engine 拥有 header 与
// epoch 初始化；槽字段归槽主）。epoch 每次引擎启动 ++：段复用（不 unlink）时
// 老映射的客户端直接看到 epoch 变化→重注册；段重建（版本/尺寸不兼容）时客户
// 端靠 writer/mixer 心跳超时探活 + 重新 open 按 inode 判别换段。
//
// 布局为 native-endian（板为 LE）、自然对齐；两端共用本头文件并 static_assert
// 关键偏移，防 ABI 漂移。
#ifndef VOICE_SHM_H
#define VOICE_SHM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VSHM_PROTO_VERSION 1u

/* 心跳/超时常量（两端一致） */
#define VSHM_HB_GRACE_MS 2000u   /* 心跳陈旧到此值：读端开始主动探测（reopen 换段判断） */
#define VSHM_HB_DEAD_MS  5000u   /* 心跳陈旧到此值：插件报 POLLERR（引擎真死） */
#define VSHM_REAP_HB_MS  15000u  /* 引擎侧：无论 pid 存活与否强制回收槽 */

/* ---------------- mic 段：引擎单写、应用多读 ---------------- */
#define VSHM_MIC_MAGIC    0x31524D56u /* "VMR1" */
#define VSHM_MIC_READERS  4u
#define VSHM_MIC_LOG2     13u        /* 8192 帧 @16k = 512ms */
#define VSHM_MIC_FRAMES   (1u << VSHM_MIC_LOG2)
#define VSHM_MIC_RATE     16000u

typedef struct {
    uint32_t used;       /* 0=空闲 1=占用；读者 CAS 0->1 抢占，引擎 epoch 初始化时清零 */
    uint32_t pid;        /* 读者 pid（引擎回收时 liveness 探测用，advisory） */
    uint64_t read_seq;   /* 该读者已消费帧数（读者独写） */
    uint64_t hb_ms;      /* 读者心跳 monotonic ms（读者独写） */
    uint64_t dropped;    /* 该读者追赶丢弃帧数（读者独写，诊断） */
} vshm_mic_slot_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;   /* VSHM_MIC_RATE */
    uint32_t channels;      /* 1 */
    uint32_t format_s16le;  /* 恒 1 */
    uint32_t frame_bytes;   /* 2 */
    uint32_t log2_frames;   /* VSHM_MIC_LOG2 */
    uint32_t n_slots;       /* VSHM_MIC_READERS */
    uint32_t epoch;         /* 引擎每次启动 ++；客户端见变即重注册 */
    uint32_t pad0;
    uint64_t write_seq;     /* 累计发布帧数（引擎独写；RELEASE 发布数据） */
    uint64_t writer_hb_ms;  /* 引擎心跳（每次发布刷新） */
    uint64_t max_lag;       /* 诊断：引擎观测到的最大读者落后帧数 */
    uint64_t pad1[5];
    vshm_mic_slot_t readers[VSHM_MIC_READERS];
    int16_t data[VSHM_MIC_FRAMES]; /* 帧 seq 位于 data[seq & (FRAMES-1)] */
} vshm_mic_t;

/* ---------------- spk 段：应用多写、引擎单读混音 ---------------- */
#define VSHM_SPK_MAGIC    0x31525356u /* "VSR1" */
#define VSHM_SPK_WRITERS  8u
#define VSHM_SPK_LOG2     9u         /* 每写者 512 帧 @48k ≈ 10.7ms 写者节拍量子（IO 8ms 帧配套；演进 4096/85ms→1024/21.3ms→512/10.7ms） */
#define VSHM_SPK_FRAMES   (1u << VSHM_SPK_LOG2)
#define VSHM_SPK_RATE     48000u

typedef struct {
    uint32_t used;
    uint32_t pid;
    uint64_t write_seq;    /* 本写者已写入帧数（写者独写；RELEASE 发布数据） */
    uint64_t consumed_seq; /* 引擎已收割帧数（引擎独写；写者据此算剩余空间背压） */
    uint64_t hb_ms;        /* 写者心跳（写者独写） */
    uint64_t dropped;      /* 引擎追赶丢弃帧数（引擎独写，诊断） */
} vshm_spk_slot_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;   /* VSHM_SPK_RATE */
    uint32_t channels;
    uint32_t format_s16le;
    uint32_t frame_bytes;
    uint32_t log2_frames;   /* VSHM_SPK_LOG2 */
    uint32_t n_slots;       /* VSHM_SPK_WRITERS */
    uint32_t epoch;         /* 引擎每次启动 ++；写者见变即重注册 */
    uint32_t pad0;
    uint64_t mixer_hb_ms;   /* 引擎收割线程心跳（每周期刷新，包括无写者时） */
    uint64_t mixed_frames;  /* 引擎混音输出总帧数（诊断） */
    uint64_t pad1[6];
    vshm_spk_slot_t writers[VSHM_SPK_WRITERS];
    int16_t data[VSHM_SPK_WRITERS][VSHM_SPK_FRAMES]; /* 槽 i 帧 seq 位于 data[i][seq & (FRAMES-1)] */
} vshm_spk_t;

/* ---------------- 布局自检（两端编译期一致） ---------------- */
#if defined(__cplusplus)
#define VSHM_SA(c, m) static_assert(c, m)
#else
#define VSHM_SA(c, m) _Static_assert(c, m)
#endif
VSHM_SA(sizeof(vshm_mic_slot_t) == 32, "mic slot size");
VSHM_SA(sizeof(vshm_spk_slot_t) == 40, "spk slot size");
VSHM_SA(offsetof(vshm_mic_t, write_seq) % 8 == 0, "mic write_seq align");
VSHM_SA(offsetof(vshm_mic_t, readers) % 8 == 0, "mic readers align");
VSHM_SA(offsetof(vshm_mic_t, readers) == 104, "mic readers offset");
VSHM_SA(offsetof(vshm_mic_t, data) == 104 + 32 * VSHM_MIC_READERS, "mic data offset");
VSHM_SA(sizeof(vshm_mic_t) == 104 + 32 * VSHM_MIC_READERS + 2 * VSHM_MIC_FRAMES, "mic seg size");
VSHM_SA(offsetof(vshm_spk_t, writers) == 104, "spk writers offset");
VSHM_SA(offsetof(vshm_spk_t, data) == 104 + 40 * VSHM_SPK_WRITERS, "spk data offset");
VSHM_SA(sizeof(vshm_spk_t) == 104 + 40 * VSHM_SPK_WRITERS + 2 * VSHM_SPK_WRITERS * VSHM_SPK_FRAMES, "spk seg size");

#ifdef __cplusplus
}
#endif

#endif /* VOICE_SHM_H */
