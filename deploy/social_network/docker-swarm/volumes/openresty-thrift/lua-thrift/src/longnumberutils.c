#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <inttypes.h>

const char * LONG_NUM_TYPE = "__thrift_longnumber";
int64_t lualongnumber_checklong(lua_State *L, int index) {
  switch (lua_type(L, index)) {
    case LUA_TNUMBER:
      return (int64_t)lua_tonumber(L, index);
    case LUA_TSTRING:
      return atoll(lua_tostring(L, index));
    default:
      return *((int64_t *)luaL_checkudata(L, index, LONG_NUM_TYPE));
  }
}

int64_t * lualongnumber_pushlong(lua_State *L, int64_t *val) {
  int64_t *data = (int64_t *)lua_newuserdata(L, sizeof(int64_t));
  luaL_getmetatable(L, LONG_NUM_TYPE);
  lua_setmetatable(L, -2);
  if (val) {
    *data = *val;
  }
  return data;
}

