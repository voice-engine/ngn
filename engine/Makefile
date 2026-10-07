# voice-engine — NanoPi Air (sun8i H3 / cortex-a7) 板上原生编译
# 板上用法: make            # 自动检测 /home/i/ngn/algo/libvoice_algo.so，缺席则 -DVA_MOCK
#          make mock       # 强制 -DVA_MOCK 桩（voice_engine_mock，测试用）: AEC out=mic / BF ch0 / DOA 0
ALGO_DIR ?= /home/i/ngn/algo
CXX      ?= g++
BASEFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread -I..
ARCHFLAGS ?= -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard   # 板上默认；x86（CI）: make ARCHFLAGS=
LDLIBS    = -lasound -lm -pthread

HDRS = ../shm_voice.h va_mock.h

ifeq ($(wildcard $(ALGO_DIR)/libvoice_algo.so),)
$(info [algo] $(ALGO_DIR)/libvoice_algo.so 不存在 → -DVA_MOCK 桩: AEC out=mic / BF ch0 / DOA 0)
ALGOFLAGS = -DVA_MOCK
SRCS = main_engine.cpp va_mock.cpp
else
$(info [algo] 链接真库 $(ALGO_DIR)/libvoice_algo.so)
ALGOFLAGS = -I$(ALGO_DIR)
SRCS = main_engine.cpp
LDLIBS += -L$(ALGO_DIR) -lvoice_algo -Wl,-rpath,$(ALGO_DIR)
endif

BIN = voice_engine
MOCK_BIN = voice_engine_mock

all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(CXX) $(BASEFLAGS) $(ARCHFLAGS) $(ALGOFLAGS) -o $@ $(SRCS) $(LDLIBS)

mock: $(MOCK_BIN)

$(MOCK_BIN): main_engine.cpp va_mock.cpp $(HDRS)
	$(CXX) $(BASEFLAGS) $(ARCHFLAGS) -DVA_MOCK -o $@ main_engine.cpp va_mock.cpp $(LDLIBS)

clean:
	rm -f $(BIN) $(MOCK_BIN)

.PHONY: all clean mock
