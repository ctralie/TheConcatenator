# https://emscripten.org/docs/tools_reference/settings_reference.html
all: particle.cpp
	emcc --bind \
		-s WASM=1\
		-s ALLOW_MEMORY_GROWTH=1\
		-s EXPORTED_FUNCTIONS="['_malloc']"\
		-s ENVIRONMENT="web,worker"\
		-s SINGLE_FILE=1\
		-o particle.js particle.cpp

#		-s USE_ES6_IMPORT_META=0\
		-s EXPORT_ES6=1\
		-s MODULARIZE=1\

clean:
	rm -f particle.js particle.wasm
