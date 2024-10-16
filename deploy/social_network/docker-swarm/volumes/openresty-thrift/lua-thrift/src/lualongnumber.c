#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>

extern const char * LONG_NUM_TYPE;
extern int64_t lualongnumber_checklong(lua_State *L, int index);
extern int64_t lualongnumber_pushlong(lua_State *L, int64_t *val);


static void l_serialize(char *buf, int len, int64_t val) {
  snprintf(buf, len, "%"PRId64, val);
}

static int64_t l_deserialize(const char *buf) {
  int64_t data;
  int rv;
  if (strstr(buf, "0x") == buf) {
    rv = sscanf(buf, "%"PRIx64, &data);
  } else {
    rv = sscanf(buf, "%"PRId64, &data);
  }
  if (rv == 1) {
    return data;
  }
  return 0;
}


static int l_new(lua_State *L) {
  int64_t val;
  const char *str = NULL;
  if (lua_type(L, 1) == LUA_TSTRING) {
    str = lua_tostring(L, 1);
    val = l_deserialize(str);
  } else if (lua_type(L, 1) == LUA_TNUMBER) {
    val = (int64_t)lua_tonumber(L, 1);
    str = (const char *)1;
  }
  lualongnumber_pushlong(L, (str ? &val : NULL));
  return 1;
}

static int l_add(lua_State *L) {
  int64_t a, b, c;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  c = a + b;
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_div(lua_State *L) {
  int64_t a, b, c;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  c = a / b;
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_eq(lua_State *L) {
  int64_t a, b;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  lua_pushboolean(L, (a == b ? 1 : 0));
  return 1;
}

static int l_gc(lua_State *L) {
  lua_pushnil(L);
  lua_setmetatable(L, 1);
  return 0;
}

static int l_lt(lua_State *L) {
  int64_t a, b;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  lua_pushboolean(L, (a < b ? 1 : 0));
  return 1;
}

static int l_le(lua_State *L) {
  int64_t a, b;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  lua_pushboolean(L, (a <= b ? 1 : 0));
  return 1;
}

static int l_mod(lua_State *L) {
  int64_t a, b, c;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  c = a % b;
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_mul(lua_State *L) {
  int64_t a, b, c;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  c = a * b;
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_pow(lua_State *L) {
  long double a, b;
  int64_t c;
  a = (long double)lualongnumber_checklong(L, 1);
  b = (long double)lualongnumber_checklong(L, 2);
  c = (int64_t)pow(a, b);
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_sub(lua_State *L) {
  int64_t a, b, c;
  a = lualongnumber_checklong(L, 1);
  b = lualongnumber_checklong(L, 2);
  c = a - b;
  lualongnumber_pushlong(L, &c);
  return 1;
}

static int l_tostring(lua_State *L) {
  int64_t a;
  char str[256];
  l_serialize(str, 256, lualongnumber_checklong(L, 1));
  lua_pushstring(L, str);
  return 1;
}

static int l_unm(lua_State *L) {
  int64_t a, c;
  a = lualongnumber_checklong(L, 1);
  c = -a;
  lualongnumber_pushlong(L, &c);
  return 1;
}


static const luaL_Reg methods[] = {
  {"__add", l_add},
  {"__div", l_div},
  {"__eq", l_eq},
  {"__gc", l_gc},
  {"__lt", l_lt},
  {"__le", l_le},
  {"__mod", l_mod},
  {"__mul", l_mul},
  {"__pow", l_pow},
  {"__sub", l_sub},
  {"__tostring", l_tostring},
  {"__unm", l_unm},
  {NULL, NULL},
};

static const luaL_Reg funcs[] = {
  {"new", l_new},
  {NULL, NULL}
};


static void set_methods(lua_State *L,
  const char *metatablename,
  const struct luaL_Reg *methods) {
  luaL_getmetatable(L, metatablename);
  for (; methods->name; methods++) {
    lua_pushstring(L, methods->name);
    lua_pushcfunction(L, methods->func);
    lua_rawset(L, -3);
  }
  lua_pop(L, 1);
}

LUALIB_API int luaopen_liblualongnumber(lua_State *L) {
  luaL_newmetatable(L, LONG_NUM_TYPE);
  lua_pop(L, 1);
  set_methods(L, LONG_NUM_TYPE, methods);

  luaL_register(L, "liblualongnumber", funcs);
  return 1;
}
