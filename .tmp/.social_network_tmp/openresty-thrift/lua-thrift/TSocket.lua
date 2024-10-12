
local TTransport = require 'TTransport'
local TTransportException = TTransport.TTransportException
local TTransportBase = TTransport.TTransportBase
local Thrift = require 'Thrift'
local ttype = Thrift.ttype
local terror = Thrift.terror

local TSocketBase = TTransportBase:new{
  __type = 'TSocketBase',
  timeout = 10000,
  host = 'localhost',
  port = 9090,
  handle
}

function TSocketBase:close()
  if self.handle then
    self.handle:close()
    self.handle = nil
  end
end

function TSocketBase:setKeepAlive(timeout, size)
  if self.handle then
    self.handle:setkeepalive(timeout, size)
  end
end

function TSocketBase:getSocketInfo()
  if self.handle then
    return self.handle:getsockinfo()
  end
  terror(TTransportException:new{errorCode = TTransportException.NOT_OPEN})
end

function TSocketBase:setTimeout(timeout)
  if timeout and ttype(timeout) == 'number' then
    if self.handle then
      self.handle:settimeout(timeout)
    end
    self.timeout = timeout
  end
end

local TSocket = TSocketBase:new{
  __type = 'TSocket',
  host = 'localhost',
  port = 9090
}

function TSocket:isOpen()
  if self.handle then
    return true
  end
  return false
end

function TSocket:open()
  if not self.handle then
    self.handle = ngx.socket.tcp()
    self.handle:settimeout(self.timeout)
  end
  local ok, err = self.handle:connect(self.host, self.port)
  if not ok then
    terror(TTransportException:new{
      message = 'Could not connect to ' .. self.host .. ':' .. self.port
        .. ' (' .. err .. ')'
    })
  end
end

function TSocket:read(len)
  local buf = self.handle:receive(len)
  if not buf or string.len(buf) ~= len then
    terror(TTransportException:new{errorCode = TTransportException.UNKNOWN})
  end
  return buf
end

function TSocket:write(buf)
  self.handle:send(buf)
end

function TSocket:flush()
end

local TServerSocket = TSocketBase:new{
  __type = 'TServerSocket',
  host = 'localhost',
  port = 9090
}

function TServerSocket:listen()
  if self.handle then
    self:close()
  end

  local sock, err = luasocket.create(self.host, self.port)
  if not err then
    self.handle = sock
  else
    terror(err)
  end
  self.handle:settimeout(self.timeout)
  self.handle:listen()
end


function TServerSocket:accept()
  local client, err = self.handle:accept()
  if err then
    terror(err)
  end
  return TSocket:new({handle = client})
end

return TSocket


