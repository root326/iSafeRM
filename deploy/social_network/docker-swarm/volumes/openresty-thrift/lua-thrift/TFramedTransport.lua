local TTransport = require 'TTransport'
local libluabpack = require 'libluabpack'
local TProtocol = require 'TProtocol'
local Thrift = require 'Thrift'
local TProtocolException = TProtocol.TProtocolException
local TTransportBase = TTransport.TTransportBase
local TTransportFactoryBase = TTransport.TTransportFactoryBase
local ttype = Thrift.ttype
local terror = Thrift.terror


local TFramedTransport = TTransportBase:new{
  __type = 'TFramedTransport',
  doRead = true,
  doWrite = true,
  wBuf = '',
  rBuf = ''
}

function TFramedTransport:new(obj)
  if ttype(obj) ~= 'table' then
    error(ttype(self) .. 'must be initialized with a table')
  end

  if not obj.trans then
    error('You must provide ' .. ttype(self) .. ' with a trans')
  end

  return TTransportBase.new(self, obj)
end

function TFramedTransport:isOpen()
  return self.trans:isOpen()
end

function TFramedTransport:open()
  return self.trans:open()
end

function TFramedTransport:close()
  return self.trans:close()
end

function TFramedTransport:read(len)
  if string.len(self.rBuf) == 0 then
    self:__readFrame()
  end

  if self.doRead == false then
    return self.trans:read(len)
  end

  if len > string.len(self.rBuf) then
    local val = self.rBuf
    self.rBuf = ''
    return val
  end

  local val = string.sub(self.rBuf, 0, len)
  self.rBuf = string.sub(self.rBuf, len+1)
  return val
end

function TFramedTransport:__readFrame()
  local buf = self.trans:readAll(4)
  local frame_len = libluabpack.bunpack('i', buf)
  self.rBuf = self.trans:readAll(frame_len)
end


function TFramedTransport:write(buf, len)
  if self.doWrite == false then
    return self.trans:write(buf, len)
  end

  if len and len < string.len(buf) then
    buf = string.sub(buf, 0, len)
  end
  self.wBuf = self.wBuf .. buf
end

function TFramedTransport:flush()
  if self.doWrite == false then
    return self.trans:flush()
  end

  local tmp = self.wBuf
  self.wBuf = ''
  local frame_len_buf = libluabpack.bpack("i", string.len(tmp))
  self.trans:write(frame_len_buf)
  self.trans:write(tmp)
  self.trans:flush()
end

local TFramedTransportFactory = TTransportFactoryBase:new{
  __type = 'TFramedTransportFactory'
}
function TFramedTransportFactory:getTransport(trans)
  if not trans then
    terror(TProtocolException:new{
      message = 'Must supply a transport to ' .. ttype(self)
    })
  end
  return TFramedTransport:new{trans = trans}
end

return TFramedTransport
