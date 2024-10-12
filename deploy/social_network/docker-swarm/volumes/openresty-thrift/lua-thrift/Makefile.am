AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = .

lib_LTLIBRARIES = \
			libluasocket.la \
			libluabpack.la \
			libluabitwise.la \
			liblualongnumber.la

libluasocket_la_SOURCES = \
			src/luasocket.c \
			src/usocket.c

nobase_include_HEADERS = src/socket.h

libluasocket_la_CPPFLAGS = $(AM_CPPFLAGS) $(LUA_INCLUDE) -DLUA_COMPAT_MODULE
libluasocket_la_LDFLAGS = $(AM_LDFLAGS)
libluasocket_la_LIBADD = $(LUA_LIB) -lm

libluabpack_la_SOURCES = src/luabpack.c

libluabpack_la_CPPFLAGS = $(AM_CPPFLAGS) $(LUA_INCLUDE) -DLUA_COMPAT_MODULE
libluabpack_la_LDFLAGS = $(AM_LDFLAGS)
libluabpack_la_LIBADD = liblualongnumber.la $(LUA_LIB) -lm

libluabitwise_la_SOURCES = src/luabitwise.c

libluabitwise_la_CPPFLAGS = $(AM_CPPFLAGS) $(LUA_INCLUDE) -DLUA_COMPAT_MODULE
libluabitwise_la_LDFLAGS = $(AM_LDFLAGS)
libluabitwise_la_LIBADD = $(LUA_LIB) -lm

liblualongnumber_la_SOURCES = \
		src/lualongnumber.c \
		src/longnumberutils.c

liblualongnumber_la_CPPFLAGS = $(AM_CPPFLAGS) $(LUA_INCLUDE) -DLUA_COMPAT_MODULE
liblualongnumber_la_LDFLAGS = $(AM_LDFLAGS)
liblualongnumber_la_LIBADD = $(LUA_LIB) -lm

EXTRA_DIST = \
		coding_standards.md \
		TBinaryProtocol.lua \
		TBufferedTransport.lua \
		TCompactProtocol.lua \
		TFramedTransport.lua \
		Thrift.lua \
		THttpTransport.lua \
		TJsonProtocol.lua \
		TMemoryBuffer.lua \
		TProtocol.lua \
		TServer.lua \
		TSocket.lua \
		TTransport.lua
