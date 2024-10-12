
local TTransport = require 'TTransport'
local Thrift = require 'Thrift'
local TTransportException = TTransport.TTransportException
local TTransportBase = TTransport.TTransportBase
local terror = Thrift.terror

local TMemoryBuffer = TTransportBase:new{
  __type = 'TMemoryBuffer',
  buffer = '',
  bufferSize = 1024,
  wPos = 0,
  rPos = 0
}
function TMemoryBuffer:isOpen()
  return 1
end
function TMemoryBuffer:open() end
function TMemoryBuffer:close() end

function TMemoryBuffer:peak()
  return self.rPos < self.wPos
end

function TMemoryBuffer:getBuffer()
  return self.buffer
end

function TMemoryBuffer:resetBuffer(buf)
  if buf then
    self.buffer = buf
    self.bufferSize = string.len(buf)
  else
    self.buffer = ''
    self.bufferSize = 1024
  end
  self.wPos = string.len(buf)
  self.rPos = 0
end

function TMemoryBuffer:available()
  return self.wPos - self.rPos
end

function TMemoryBuffer:read(len)
  local avail = self:available()
  if avail == 0 then
    return ''
  end

  if avail < len then
    len = avail
  end

  local val = string.sub(self.buffer, self.rPos + 1, self.rPos + len)
  self.rPos = self.rPos + len
  return val
end

function TMemoryBuffer:readAll(len)
  local avail = self:available()

  if avail < len then
    local msg = string.format('Attempt to readAll(%d) found only %d available',
                              len, avail)
    terror(TTransportException:new{message = msg})
  end
  return self:read(len)
end

function TMemoryBuffer:write(buf)
  self.buffer = self.buffer .. buf
  self.wPos = self.wPos + string.len(buf)
end

function TMemoryBuffer:flush() end
