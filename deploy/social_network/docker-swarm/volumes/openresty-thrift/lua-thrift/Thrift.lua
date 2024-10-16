package.cpath = package.cpath .. ';bin/?.so'
local function ttype(obj)
  if type(obj) == 'table' and
    obj.__type and
    type(obj.__type) == 'string' then
      return obj.__type
  end
  return type(obj)
end

local function terror(e)
  if e and e.__tostring then
    error(e:__tostring())
    return
  end
  error(e)
end

local function ttable_size(t)
  local count = 0
  for k, v in pairs(t) do
    count = count + 1
  end
  return count
end

local version = 1.0

local TType = {
  STOP   = 0,
  VOID   = 1,
  BOOL   = 2,
  BYTE   = 3,
  I08    = 3,
  DOUBLE = 4,
  I16    = 6,
  I32    = 8,
  I64    = 10,
  STRING = 11,
  UTF7   = 11,
  STRUCT = 12,
  MAP    = 13,
  SET    = 14,
  LIST   = 15,
  UTF8   = 16,
  UTF16  = 17
}

local TMessageType = {
  CALL  = 1,
  REPLY = 2,
  EXCEPTION = 3,
  ONEWAY = 4
}

local function __tobj_index(self, key)
  local v = rawget(self, key)
  if v ~= nil then
    return v
  end

  local p = rawget(self, '__parent')
  if p then
    return __tobj_index(p, key)
  end

  return nil
end

local __TObject = {
  __type = '__TObject',
  __mt = {
    __index = __tobj_index
  }
}
function __TObject:new(init_obj)
  local obj = {}
  if ttype(obj) == 'table' then
    obj = init_obj
  end

  obj.__parent = self
  setmetatable(obj, __TObject.__mt)
  return obj
end

local function thrift_print_r(t)
  local ret = ''
  local ltype = type(t)
  if (ltype == 'table') then
    ret = ret .. '{ '
    for key,value in pairs(t) do
      ret = ret .. tostring(key) .. '=' .. thrift_print_r(value) .. ' '
    end
    ret = ret .. '}'
  elseif ltype == 'string' then
    ret = ret .. "'" .. tostring(t) .. "'"
  else
    ret = ret .. tostring(t)
  end
  return ret
end

local TException = __TObject:new{
  message,
  errorCode,
  __type = 'TException'
}
function TException:__tostring()
  if self.message then
    return string.format('%s: %s', self.__type, self.message)
  else
    local message
    if self.errorCode and self.__errorCodeToString then
      message = string.format('%d: %s', self.errorCode, self:__errorCodeToString())
    else
      message = thrift_print_r(self)
    end
    return string.format('%s:%s', self.__type, message)
  end
end

local TApplicationException = TException:new{
  UNKNOWN                 = 0,
  UNKNOWN_METHOD          = 1,
  INVALID_MESSAGE_TYPE    = 2,
  WRONG_METHOD_NAME       = 3,
  BAD_SEQUENCE_ID         = 4,
  MISSING_RESULT          = 5,
  INTERNAL_ERROR          = 6,
  PROTOCOL_ERROR          = 7,
  INVALID_TRANSFORM       = 8,
  INVALID_PROTOCOL        = 9,
  UNSUPPORTED_CLIENT_TYPE = 10,
  errorCode               = 0,
  __type = 'TApplicationException'
}

function TApplicationException:__errorCodeToString()
  if self.errorCode == self.UNKNOWN_METHOD then
    return 'Unknown method'
  elseif self.errorCode == self.INVALID_MESSAGE_TYPE then
    return 'Invalid message type'
  elseif self.errorCode == self.WRONG_METHOD_NAME then
    return 'Wrong method name'
  elseif self.errorCode == self.BAD_SEQUENCE_ID then
    return 'Bad sequence ID'
  elseif self.errorCode == self.MISSING_RESULT then
    return 'Missing result'
  elseif self.errorCode == self.INTERNAL_ERROR then
    return 'Internal error'
  elseif self.errorCode == self.PROTOCOL_ERROR then
    return 'Protocol error'
  elseif self.errorCode == self.INVALID_TRANSFORM then
    return 'Invalid transform'
  elseif self.errorCode == self.INVALID_PROTOCOL then
    return 'Invalid protocol'
  elseif self.errorCode == self.UNSUPPORTED_CLIENT_TYPE then
    return 'Unsupported client type'
  else
    return 'Default (unknown)'
  end
end

function TException:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 1 then
      if ftype == TType.STRING then
        self.message = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 2 then
      if ftype == TType.I32 then
        self.errorCode = iprot:readI32()
      else
        iprot:skip(ftype)
      end
    else
      iprot:skip(ftype)
    end
    iprot:readFieldEnd()
  end
  iprot:readStructEnd()
end

function TException:write(oprot)
  oprot:writeStructBegin('TApplicationException')
  if self.message then
    oprot:writeFieldBegin('message', TType.STRING, 1)
    oprot:writeString(self.message)
    oprot:writeFieldEnd()
  end
  if self.errorCode then
    oprot:writeFieldBegin('type', TType.I32, 2)
    oprot:writeI32(self.errorCode)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local __TClient = __TObject:new{
  __type = '__TClient',
  _seqid = 0
}
function __TClient:new(obj)
  if ttype(obj) ~= 'table' then
    error('TClient must be initialized with a table')
  end

  if obj.protocol then
    obj.iprot = obj.protocol
    obj.oprot = obj.protocol
    obj.protocol = nil
  elseif not obj.iprot then
    error('You must provide ' .. ttype(self) .. ' with an iprot')
  end
  if not obj.oprot then
    obj.oprot = obj.iprot
  end

  return __TObject.new(self, obj)
end

function __TClient:close()
  self.iprot.trans:close()
  self.oprot.trans:close()
end

local __TProcessor = __TObject:new{
  __type = '__TProcessor'
}
function __TProcessor:new(obj)
  if ttype(obj) ~= 'table' then
    error('TProcessor must be initialized with a table')
  end

  if not obj.handler then
    error('You must provide ' .. ttype(self) .. ' with a handler')
  end

  return __TObject.new(self, obj)
end

return {
    TType=TType,
    TMessageType=TMessageType,
    __TObject=__TObject,
    TException=TException,
    TApplicationException=TApplicationException,
    __TClient=__TClient,
    __TProcessor=__TProcessor,
    ttype=ttype,
    terror=terror,
    ttable_size=ttable_size
}
