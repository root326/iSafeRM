

local TSocket = require 'TSocket'
local Thrift = require 'Thrift'
local TTransport = require 'TTransport'
local TTransportException = TTransport.TTransportException
local ttype = Thrift.ttype
local terror = Thrift.terror

local TSocketSSL = TSocket:new{
  __type = 'TSocketSSL',
  host = 'localhost',
  port = 9090
}

function TSocketSSL:open()
  ngx.log(ngx.INFO, "ssl open called")
  if not self.handle then
    self.handle = ngx.socket.tcp()
    self.handle:settimeout(self.timeout)
  end
  ngx.log(ngx.INFO, "ssl start to connect")
  local ok, err = self.handle:connect(self.host, self.port)
  if not ok then
    terror(TTransportException:new{
      message = 'Could not connect to ' .. self.host .. ':' .. self.port
        .. ' (' .. err .. ')'
    })
  end
  ngx.log(ngx.INFO, "ssl handshake start")
  local session, err = self.handle:sslhandshake(nil, nil, true)
  if not session then
    terror(TTransportException:new{
      message = 'failed to do hand shake with ' .. self.host .. ':' .. self.port
        .. ' (' .. err .. ')'
    })
  end
  ngx.log(ngx.INFO, "ssl handshake end")
end

return TSocketSSL
