local Object = require 'Object'
local RpcClientFactory = require 'RpcClientFactory'
local ngx = ngx
local GenericObjectPool = Object:new({
    __type = 'GenericObjectPool',
    maxTotal = 100,
    maxIdleTime = 10000,
    timeout = 10000
    })
function GenericObjectPool:init(conf)
end

function GenericObjectPool:connection(thriftClient,ip,port)
    local ssl = ngx.shared.config:get("ssl")
    local client = RpcClientFactory:createClient(thriftClient,ip,port,self.timeout,ssl)
    return client
end
function GenericObjectPool:returnConnection(client)
    if(client ~= nil)then
        if (client.iprot.trans.trans:isOpen())then
            client.iprot.trans.trans:setKeepAlive(self.maxIdleTime, self.maxTotal)
        else
            ngx.log(ngx.ERR,"return rpc client fail, socket close.")
        end
    end
end

function GenericObjectPool:setMaxTotal(maxTotal)
    self.maxTotal = maxTotal
end

function GenericObjectPool:setmaxIdleTime(maxIdleTime)
    self.maxIdleTime = maxIdleTime
end

function GenericObjectPool:setTimeout(timeout)
    self.timeout = timeout
end

function GenericObjectPool:clear()

end
function GenericObjectPool:remove()

end
return GenericObjectPool
