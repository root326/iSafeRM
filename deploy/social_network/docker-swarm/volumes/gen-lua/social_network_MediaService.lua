
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

local UploadMedia_args = __TObject:new{
  req_id,
  media_types,
  media_ids,
  carrier
}

function UploadMedia_args:read(iprot)
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
      if ftype == TType.LIST then
        self.media_types = {}
        local _etype325, _size322 = iprot:readListBegin()
        for _i=1,_size322 do
          local _elem326 = iprot:readString()
          table.insert(self.media_types, _elem326)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.LIST then
        self.media_ids = {}
        local _etype330, _size327 = iprot:readListBegin()
        for _i=1,_size327 do
          local _elem331 = iprot:readI64()
          table.insert(self.media_ids, _elem331)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype333, _vtype334, _size332 = iprot:readMapBegin()
        for _i=1,_size332 do
          local _key336 = iprot:readString()
          local _val337 = iprot:readString()
          self.carrier[_key336] = _val337
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

function UploadMedia_args:write(oprot)
  oprot:writeStructBegin('UploadMedia_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.media_types ~= nil then
    oprot:writeFieldBegin('media_types', TType.LIST, 2)
    oprot:writeListBegin(TType.STRING, #self.media_types)
    for _,iter338 in ipairs(self.media_types) do
      oprot:writeString(iter338)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.media_ids ~= nil then
    oprot:writeFieldBegin('media_ids', TType.LIST, 3)
    oprot:writeListBegin(TType.I64, #self.media_ids)
    for _,iter339 in ipairs(self.media_ids) do
      oprot:writeI64(iter339)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter340,viter341 in pairs(self.carrier) do
      oprot:writeString(kiter340)
      oprot:writeString(viter341)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local UploadMedia_result = __TObject:new{
  se
}

function UploadMedia_result:read(iprot)
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

function UploadMedia_result:write(oprot)
  oprot:writeStructBegin('UploadMedia_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local MediaServiceClient = __TObject.new(__TClient, {
  __type = 'MediaServiceClient'
})

function MediaServiceClient:UploadMedia(req_id, media_types, media_ids, carrier)
  self:send_UploadMedia(req_id, media_types, media_ids, carrier)
  self:recv_UploadMedia(req_id, media_types, media_ids, carrier)
end

function MediaServiceClient:send_UploadMedia(req_id, media_types, media_ids, carrier)
  self.oprot:writeMessageBegin('UploadMedia', TMessageType.CALL, self._seqid)
  local args = UploadMedia_args:new{}
  args.req_id = req_id
  args.media_types = media_types
  args.media_ids = media_ids
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function MediaServiceClient:recv_UploadMedia(req_id, media_types, media_ids, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadMedia_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end
local MediaServiceIface = __TObject:new{
  __type = 'MediaServiceIface'
}


local MediaServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'MediaServiceProcessor'
})

function MediaServiceProcessor:process(iprot, oprot, server_ctx)
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

function MediaServiceProcessor:process_UploadMedia(seqid, iprot, oprot, server_ctx)
  local args = UploadMedia_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadMedia_result:new{}
  local status, res = pcall(self.handler.UploadMedia, self.handler, args.req_id, args.media_types, args.media_ids, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadMedia', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

return MediaServiceClient