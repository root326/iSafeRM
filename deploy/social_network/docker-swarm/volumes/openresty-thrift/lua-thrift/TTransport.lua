
local Thrift = require('Thrift')
local __TObject = Thrift.__TObject
local TException = Thrift.TException
local terror = Thrift.terror

local TTransportException = TException:new {
  UNKNOWN             = 0,
  NOT_OPEN            = 1,
  ALREADY_OPEN        = 2,
  TIMED_OUT           = 3,
  END_OF_FILE         = 4,
  INVALID_FRAME_SIZE  = 5,
  INVALID_TRANSFORM   = 6,
  INVALID_CLIENT_TYPE = 7,
  errorCode        = 0,
  __type = 'TTransportException'
}

function TTransportException:__errorCodeToString()
  if self.errorCode == self.NOT_OPEN then
    return 'Transport not open'
  elseif self.errorCode == self.ALREADY_OPEN then
    return 'Transport already open'
  elseif self.errorCode == self.TIMED_OUT then
    return 'Transport timed out'
  elseif self.errorCode == self.END_OF_FILE then
    return 'End of file'
  elseif self.errorCode == self.INVALID_FRAME_SIZE then
    return 'Invalid frame size'
  elseif self.errorCode == self.INVALID_TRANSFORM then
    return 'Invalid transform'
  elseif self.errorCode == self.INVALID_CLIENT_TYPE then
    return 'Invalid client type'
  else
    return 'Default (unknown)'
  end
end

local TTransportBase = __TObject:new{
  __type = 'TTransportBase'
}

function TTransportBase:isOpen() end
function TTransportBase:open() end
function TTransportBase:close() end
function TTransportBase:read(len) end
function TTransportBase:readAll(len)
  local buf, have, chunk = '', 0
  while have < len do
    chunk = self:read(len - have)
    have = have + string.len(chunk)
    buf = buf .. chunk

    if string.len(chunk) == 0 then
      terror(TTransportException:new{
        errorCode = TTransportException.END_OF_FILE
      })
    end
  end
  return buf
end
function TTransportBase:write(buf) end
function TTransportBase:flush() end

local TServerTransportBase = __TObject:new{
  __type = 'TServerTransportBase'
}
function TServerTransportBase:listen() end
function TServerTransportBase:accept() end
function TServerTransportBase:close() end

local TTransportFactoryBase = __TObject:new{
  __type = 'TTransportFactoryBase'
}
function TTransportFactoryBase:getTransport(trans)
  return trans
end

return {
  TTransportException=TTransportException,
  TTransportBase=TTransportBase,
  TTransportFactoryBase=TTransportFactoryBase
}
