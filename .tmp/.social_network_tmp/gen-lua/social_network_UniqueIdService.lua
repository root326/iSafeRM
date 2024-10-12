
local Thrift = require 'Thrift'
local TType = Thrift.TType
local TMessageType = Thrift.TMessageType
local __TObject = Thrift.__TObject
local TApplicationException = Thrift.TApplicationException
local __TClient = Thrift.__TClient
local __TProcessor = Thrift.__TProcessor
local ttype = Thrift.ttype
local ttable_size = Thrift.ttable_size
local social_network_ttypes = require 'social_network_ttypes'
local ServiceException = social_network_ttypes.ServiceException

local UploadUniqueId_args = __TObject:new{
  req_id,
  post_type,
  carrier
}

function UploadUniqueId_args:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 1 then
      if ftype == TType.I64 then
        self.req_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 2 then
      if ftype == TType.I32 then
        self.post_type = iprot:readI32()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype19, _vtype20, _size18 = iprot:readMapBegin()
        for _i=1,_size18 do
          local _key22 = iprot:readString()
          local _val23 = iprot:readString()
          self.carrier[_key22] = _val23
        end
        iprot:readMapEnd()
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

function UploadUniqueId_args:write(oprot)
  oprot:writeStructBegin('UploadUniqueId_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.post_type ~= nil then
    oprot:writeFieldBegin('post_type', TType.I32, 2)
    oprot:writeI32(self.post_type)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter24,viter25 in pairs(self.carrier) do
      oprot:writeString(kiter24)
      oprot:writeString(viter25)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local UploadUniqueId_result = __TObject:new{
  se
}

function UploadUniqueId_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 1 then
      if ftype == TType.STRUCT then
        self.se = ServiceException:new{}
        self.se:read(iprot)
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

function UploadUniqueId_result:write(oprot)
  oprot:writeStructBegin('UploadUniqueId_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local UniqueIdServiceClient = __TObject.new(__TClient, {
  __type = 'UniqueIdServiceClient'
})

function UniqueIdServiceClient:UploadUniqueId(req_id, post_type, carrier)
  self:send_UploadUniqueId(req_id, post_type, carrier)
  self:recv_UploadUniqueId(req_id, post_type, carrier)
end

function UniqueIdServiceClient:send_UploadUniqueId(req_id, post_type, carrier)
  self.oprot:writeMessageBegin('UploadUniqueId', TMessageType.CALL, self._seqid)
  local args = UploadUniqueId_args:new{}
  args.req_id = req_id
  args.post_type = post_type
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UniqueIdServiceClient:recv_UploadUniqueId(req_id, post_type, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadUniqueId_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end
local UniqueIdServiceIface = __TObject:new{
  __type = 'UniqueIdServiceIface'
}


local UniqueIdServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'UniqueIdServiceProcessor'
})

function UniqueIdServiceProcessor:process(iprot, oprot, server_ctx)
  local name, mtype, seqid = iprot:readMessageBegin()
  local func_name = 'process_' .. name
  if not self[func_name] or ttype(self[func_name]) ~= 'function' then
    iprot:skip(TType.STRUCT)
    iprot:readMessageEnd()
    x = TApplicationException:new{
      errorCode = TApplicationException.UNKNOWN_METHOD
    }
    oprot:writeMessageBegin(name, TMessageType.EXCEPTION, seqid)
    x:write(oprot)
    oprot:writeMessageEnd()
    oprot.trans:flush()
  else
    self[func_name](self, seqid, iprot, oprot, server_ctx)
  end
end

function UniqueIdServiceProcessor:process_UploadUniqueId(seqid, iprot, oprot, server_ctx)
  local args = UploadUniqueId_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadUniqueId_result:new{}
  local status, res = pcall(self.handler.UploadUniqueId, self.handler, args.req_id, args.post_type, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadUniqueId', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

return UniqueIdServiceClient