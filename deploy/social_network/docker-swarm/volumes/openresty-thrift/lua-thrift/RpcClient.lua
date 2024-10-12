local TSocket = require "TSocket"
local TSocketSSL = require "TSocketSSL"
local TFramedTransport = require "TFramedTransport"
local TBinaryProtocol = require "TBinaryProtocol"
local Object = require "Object"

local RpcClient = Object:new({
	__type = 'RpcClient',
})

function RpcClient:init(ip,port,timeout,ssl)
	if (ssl == true) then
		socket = TSocketSSL:new{
			host = ip,
			port = port
		 }
	else
		socket = TSocket:new{
			host = ip,
			port = port
		}
	end
	socket:setTimeout(timeout)
	local transport = TFramedTransport:new{
		trans = socket
	}
	local protocol = TBinaryProtocol:new{
		trans = transport
	}
	transport:open()
	return protocol;
end
function RpcClient:createClient(thriftClient)end

return RpcClient
