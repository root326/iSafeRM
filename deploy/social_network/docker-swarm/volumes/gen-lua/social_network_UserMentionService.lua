require 'Thrift'
require 'social_network_ttypes'

UserMentionServiceClient = __TObject.new(__TClient, {
  __type = 'UserMentionServiceClient'
})

function UserMentionServiceClient:UploadUserMentions(req_id, usernames, carrier)
  self:send_UploadUserMentions(req_id, usernames, carrier)
  self:recv_UploadUserMentions(req_id, usernames, carrier)
end

function UserMentionServiceClient:send_UploadUserMentions(req_id, usernames, carrier)
  self.oprot:writeMessageBegin('UploadUserMentions', TMessageType.CALL, self._seqid)
  local args = UploadUserMentions_args:new{}
  args.req_id = req_id
  args.usernames = usernames
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UserMentionServiceClient:recv_UploadUserMentions(req_id, usernames, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadUserMentions_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end
UserMentionServiceIface = __TObject:new{
  __type = 'UserMentionServiceIface'
}


UserMentionServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'UserMentionServiceProcessor'
})

function UserMentionServiceProcessor:process(iprot, oprot, server_ctx)
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

function UserMentionServiceProcessor:process_UploadUserMentions(seqid, iprot, oprot, server_ctx)
  local args = UploadUserMentions_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadUserMentions_result:new{}
  local status, res = pcall(self.handler.UploadUserMentions, self.handler, args.req_id, args.usernames, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadUserMentions', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end


UploadUserMentions_args = __TObject:new{
  req_id,
  usernames,
  carrier
}

function UploadUserMentions_args:read(iprot)
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
        self.usernames = {}
        local _etype239, _size236 = iprot:readListBegin()
        for _i=1,_size236 do
          local _elem240 = iprot:readString()
          table.insert(self.usernames, _elem240)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype242, _vtype243, _size241 = iprot:readMapBegin() 
        for _i=1,_size241 do
          local _key245 = iprot:readString()
          local _val246 = iprot:readString()
          self.carrier[_key245] = _val246
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

function UploadUserMentions_args:write(oprot)
  oprot:writeStructBegin('UploadUserMentions_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.usernames ~= nil then
    oprot:writeFieldBegin('usernames', TType.LIST, 2)
    oprot:writeListBegin(TType.STRING, #self.usernames)
    for _,iter247 in ipairs(self.usernames) do
      oprot:writeString(iter247)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter248,viter249 in pairs(self.carrier) do
      oprot:writeString(kiter248)
      oprot:writeString(viter249)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUserMentions_result = __TObject:new{
  se
}

function UploadUserMentions_result:read(iprot)
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

function UploadUserMentions_result:write(oprot)
  oprot:writeStructBegin('UploadUserMentions_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end