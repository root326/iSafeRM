
#include <lua.h>
#include <lauxlib.h>

static int l_not(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  a = ~a;
  lua_pushnumber(L, a);
  return 1;
}

static int l_xor(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  int b = luaL_checkinteger(L, 2);
  a ^= b;
  lua_pushnumber(L, a);
  return 1;
}

static int l_and(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  int b = luaL_checkinteger(L, 2);
  a &= b;
  lua_pushnumber(L, a);
  return 1;
}

static int l_or(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  int b = luaL_checkinteger(L, 2);
  a |= b;
  lua_pushnumber(L, a);
  return 1;
}

static int l_shiftr(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  int b = luaL_checkinteger(L, 2);
  a = a >> b;
  lua_pushnumber(L, a);
  return 1;
}

static int l_shiftl(lua_State *L) {
  int a = luaL_checkinteger(L, 1);
  int b = luaL_checkinteger(L, 2);
  a = a << b;
  lua_pushnumber(L, a);
  return 1;
}

static const struct luaL_Reg funcs[] = {
  {"band", l_and},
  {"bor", l_or},
  {"bxor", l_xor},
  {"bnot", l_not},
  {"shiftl", l_shiftl},
  {"shiftr", l_shiftr},
  {NULL, NULL}
};

int luaopen_libluabitwise(lua_State *L) {
  luaL_register(L, "libluabitwise", funcs);
  return 1;
}
