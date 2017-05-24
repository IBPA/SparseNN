package = "SparseNN"
version = "scm-1"
source = {
   url = "git://github.com/ameenetemady/SparseNN",
   tag = "master"
}

description = {
   summary = "This package provides neural network modules supporting sparse data in various levels using Torch",
   homepage = "https://github.com/ameenetemady/SparseNN"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 0.1"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"  -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
