# ALSA fifo/shm plugins - NanoPi port
# Build:  make           # libasound_module_pcm_fifo.so + libasound_module_pcm_vshm.so
# Install: make install  (uses sudo; passwordless sudo confirmed on this box)

# Quiet (set to @ for a quite compile)
Q	?= @
#Q	?=@

# Build Tools
CC 	:= gcc
CFLAGS += -I.. -Wall -funroll-loops -ffast-math -fPIC -DPIC -O2 -g
LD := gcc
LDFLAGS += -Wall -shared

SND_PCM_LIBS = -lasound

# fifo 插件（旧链路 voice_mic/voice_spk）+ shm 插件（voice_mic_shm/voice_spk_shm，
# 协议 ../shm_voice.h，与引擎共用）
BINS = libasound_module_pcm_fifo.so libasound_module_pcm_vshm.so

MULTIARCH:=$(shell gcc --print-multiarch)
LIBDIR = lib/$(MULTIARCH)

# plugindir detection: pkg-config first, fall back to the Debian multiarch path
# (/usr/lib/arm-linux-gnueabihf/alsa-lib on this NanoPi; pkg-config --variable
# returns empty here but the directory is the live plugin dir - ac108 lives in it)
PLUGINDIR := $(shell pkg-config --variable=plugindir alsa)
ifeq ($(strip $(PLUGINDIR)),)
PLUGINDIR := /usr/$(LIBDIR)/alsa-lib
endif

.PHONY: all clean dep install uninstall

all: Makefile $(BINS)

dep:
	@echo DEP $@
	$(Q)for i in *.c; do $(CC) -MM $(CFLAGS) "$${i}" ; done > makefile.dep

-include makefile.dep

libasound_module_pcm_fifo.so: pcm_fifo.o
	@echo LD $@
	$(Q)$(LD) $(LDFLAGS) $< $(SND_PCM_LIBS) -o $@

libasound_module_pcm_vshm.so: pcm_shm.o
	@echo LD $@
	$(Q)$(LD) $(LDFLAGS) $< $(SND_PCM_LIBS) -o $@

%.o: %.c ../shm_voice.h
	@echo GCC $<
	$(Q)$(CC) -c $(CFLAGS) $(CPPFLAGS) $<

clean:
	@echo Cleaning...
	$(Q)rm -vf *.o *.so

install: all
	@echo Installing to $(PLUGINDIR) ...
	$(Q)sudo install -m 644 $(BINS) $(PLUGINDIR)/
	@echo Installed: $(BINS:%=$(PLUGINDIR)/%)

uninstall:
	$(Q)sudo rm -f $(BINS:%=$(PLUGINDIR)/%)
