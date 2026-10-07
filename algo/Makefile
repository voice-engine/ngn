# voice/algo — libvoice_algo（NanoPi Air / Cortex-A7 NEON 原生构建）
#
# 产物：
#   libvoice_algo.so  共享库（引擎 dlopen / 直接链）
#   libvoice_algo.a   静态库
#   test_algo         板上单测（合成数据，exit 0 = 全部通过）
#
# NEON：-mcpu=cortex-a7 -mfpu=neon-vfpv4（H2+/H3 四核 Cortex-A7，
# NEONv2 + VFPv4 硬浮点；Debian armhf 默认 hard ABI）。
# `make neon-stat` 输出开/不开 -mcpu 的向量指令条数对比（objdump -d 抽查）。

CXX      ?= g++
AR       ?= ar
BASEFLAGS ?= -std=c++17 -O3 -Wall -Wextra -fPIC
NEON     ?= -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard
CXXFLAGS := $(BASEFLAGS) $(NEON)

OBJS = aec_kf.o bf.o voice_algo.o

all: libvoice_algo.so libvoice_algo.a test_algo

aec_kf.o: aec_kf.cpp aec_kf.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ aec_kf.cpp

bf.o: bf.cpp bf.hpp fft_bf.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ bf.cpp

voice_algo.o: voice_algo.cpp voice_algo.h aec_kf.hpp bf.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ voice_algo.cpp

libvoice_algo.so: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(OBJS)

libvoice_algo.a: $(OBJS)
	$(AR) rcs $@ $(OBJS)

test_algo: test_algo.cpp libvoice_algo.a libvoice_algo.so
	$(CXX) $(CXXFLAGS) -o $@ test_algo.cpp -L. -lvoice_algo -ldl \
	    -Wl,-rpath,'$$ORIGIN'

test: test_algo
	./test_algo

# NEON 验证：同一源文件两份（带/不带 -mcpu=cortex-a7），objdump 统计
# 向量指令（v 前缀助记符，如 vld1/vst1/vmla/vmul.f32/vadd.f32 …）。
neon-stat:
	@mkdir -p .neoncheck
	$(CXX) $(BASEFLAGS) $(NEON) -c aec_kf.cpp -o .neoncheck/aec_kf_neon.o
	$(CXX) $(BASEFLAGS) -c aec_kf.cpp -o .neoncheck/aec_kf_base.o
	$(CXX) $(BASEFLAGS) $(NEON) -c bf.cpp -o .neoncheck/bf_neon.o
	$(CXX) $(BASEFLAGS) -c bf.cpp -o .neoncheck/bf_base.o
	$(CXX) $(BASEFLAGS) $(NEON) -c voice_algo.cpp -o .neoncheck/voice_algo_neon.o
	$(CXX) $(BASEFLAGS) -c voice_algo.cpp -o .neoncheck/voice_algo_base.o
	@for f in aec_kf bf voice_algo; do \
	  for v in neon base; do \
	    total=$$(objdump -d .neoncheck/$${f}_$${v}.o \
	            | grep -oE '\sv[a-z0-9]+(\.[a-z0-9]+)*[[:space:]]' | wc -l | tr -d ' '); \
	    ldst=$$(objdump -d .neoncheck/$${f}_$${v}.o \
	            | grep -cE '\sv(ld|st)[0-9]'); \
	    fma=$$(objdump -d .neoncheck/$${f}_$${v}.o \
	            | grep -cE '\sv(mla|mls|fma|fms)'); \
	    echo "$$f [$$v]: 向量指令 $$total 条 (vld/vst $$ldst, vmla/vfma $$fma)"; \
	  done; \
	done
	@echo '--- 抽查（neon 版示例指令）---'
	@objdump -d .neoncheck/aec_kf_neon.o | grep -E '\sv(ld1|vmla|vfma|vmul|vst1)' | head -8

clean:
	rm -f $(OBJS) libvoice_algo.so libvoice_algo.a test_algo
	rm -rf .neoncheck

.PHONY: all test neon-stat clean
