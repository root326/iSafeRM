
local TTransport = require 'TTransport'
local TTransportException = TTransport.TTransportException
local TTransportBase = TTransport.TTransportBase
local TTransportFactoryBase = TTransport.TTransportFactoryBase
local Thrift = require 'Thrift'
local ttype = Thrift.ttype
local terror = Thrift.terror

local TBufferedTransport = TTransportBase:new{
  __type = 'TBufferedTransport',
  rBufSize = 2048,
  wBufSize = 2048,
  wBuf = '',
  rBuf = ''
}

function TBufferedTransport:new(obj)
  if ttype(obj) ~= 'table' then
    error(ttype(self) .. 'must be initialized with a table')
  end

  if not obj.trans then
    error('You must provide ' .. ttype(self) .. ' with a trans')
  end

  return TTransportBase.new(self, obj)
end

function TBufferedTransport:isOpen()
  return self.trans:isOpen()
end

function TBufferedTransport:open()
  return self.trans:open()
end

function TBufferedTransport:close()
  return self.trans:close()
end

function TBufferedTransport:read(len)
  return self.trans:read(len)
end

function TBufferedTransport:readAll(len)
  return self.trans:readAll(len)
end

function TBufferedTransport:write(buf)
  self.wBuf = self.wBuf .. buf
  if string.len(self.wBuf) >= self.wBufSize then
    self.trans:write(self.wBuf)
    self.wBuf = ''
  end
end

function TBufferedTransport:flush()
  if string.len(self.wBuf) > 0 then
    self.trans:write(self.wBuf)
    self.wBuf = ''
  end
end

local TBufferedTransportFactory = TTransportFactoryBase:new{
  __type = 'TBufferedTransportFactory'
}

function TBufferedTransportFactory:getTransport(trans)
  if not trans then
    terror(TTransportException:new{
      message = 'Must supply a transport to ' .. ttype(self)
    })
  end
  return TBufferedTransport:new{
    trans = trans
  }
end

return TBufferedTransport
