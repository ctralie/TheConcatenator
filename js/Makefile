all: particle.cpp
	emcc -lembind -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS="['_malloc']" -o particle.js particle.cpp

clean:
	rm -f particle.js particle.wasm
