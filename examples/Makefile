all: conv_metrics conv_event

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include
CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti


# demo: CUPTI_Conv_Metrics.cu
conv_metrics : CUPTI_Conv_Metrics.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_Conv_Metrics.cu -o conv_metrics
conv_event : CUPTI_Conv_Event.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_Conv_Event.cu -o conv_event


clean:
	rm -f *.o conv_metrics conv_event
