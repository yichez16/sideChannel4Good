all: verifier generator

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include
CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti


verifier : CUPTI_verifier.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_verifier.cu -o verifier
generator : CUPTI_generator.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) CUPTI_generator.cu -o generator


clean: 
	rm -f *.o conv_metrics conv_event verifier generator
