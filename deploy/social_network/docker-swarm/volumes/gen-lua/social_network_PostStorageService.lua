require 'Thrift'
require 'social_network_ttypes'

PostStorageServiceClient = __TObject.new(__TClient, {
  __type = 'PostStorageServiceClient'
})

function PostStorageServiceClient:StorePost(req_id, post, carrier)
  self:send_StorePost(req_id, post, carrier)
  self:recv_StorePost(req_id, post, carrier)
end

function PostStorageServiceClient:send_StorePost(req_id, post, carrier)
  self.oprot:writeMessageBegin('StorePost', TMessageType.CALL, self._seqid)
  local args = StorePost_args:new{}
  args.req_id = req_id
  args.post = post
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function PostStorageServiceClient:recv_StorePost(req_id, post, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = StorePost_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end

function PostStorageServiceClient:ReadPost(req_id, post_id, carrier)
  self:send_ReadPost(req_id, post_id, carrier)
  return self:recv_ReadPost(req_id, post_id, carrier)
end

function PostStorageServiceClient:send_ReadPost(req_id, post_id, carrier)
  self.oprot:writeMessageBegin('ReadPost', TMessageType.CALL, self._seqid)
  local args = ReadPost_args:new{}
  args.req_id = req_id
  args.post_id = post_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function PostStorageServiceClient:recv_ReadPost(req_id, post_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = ReadPost_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end
PostStorageServiceIface = __TObject:new{
  __type = 'PostStorageServiceIface'
}


PostStorageServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'PostStorageServiceProcessor'
})

function PostStorageServiceProcessor:process(iprot, oprot, server_ctx)
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

function PostStorageServiceProcessor:process_StorePost(seqid, iprot, oprot, server_ctx)
  local args = StorePost_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = StorePost_result:new{}
  local status, res = pcall(self.handler.StorePost, self.handler, args.req_id, args.post, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('StorePost', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function PostStorageServiceProcessor:process_ReadPost(seqid, iprot, oprot, server_ctx)
  local args = ReadPost_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = ReadPost_result:new{}
  local status, res = pcall(self.handler.ReadPost, self.handler, args.req_id, args.post_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('ReadPost', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

StorePost_args = __TObject:new{
  req_id,
  post,
  carrier
}

function StorePost_args:read(iprot)
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
      if ftype == TType.STRUCT then
        self.post = Post:new{}
        self.post:read(iprot)
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype141, _vtype142, _size140 = iprot:readMapBegin() 
        for _i=1,_size140 do
          local _key144 = iprot:readString()
          local _val145 = iprot:readString()
          self.carrier[_key144] = _val145
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

function StorePost_args:write(oprot)
  oprot:writeStructBegin('StorePost_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.post ~= nil then
    oprot:writeFieldBegin('post', TType.STRUCT, 2)
    self.post:write(oprot)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter146,viter147 in pairs(self.carrier) do
      oprot:writeString(kiter146)
      oprot:writeString(viter147)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

StorePost_result = __TObject:new{
  se
}

function StorePost_result:read(iprot)
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

function StorePost_result:write(oprot)
  oprot:writeStructBegin('StorePost_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

ReadPost_args = __TObject:new{
  req_id,
  post_id,
  carrier
}

function ReadPost_args:read(iprot)
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
      if ftype == TType.I64 then
        self.post_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype149, _vtype150, _size148 = iprot:readMapBegin() 
        for _i=1,_size148 do
          local _key152 = iprot:readString()
          local _val153 = iprot:readString()
          self.carrier[_key152] = _val153
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

function ReadPost_args:write(oprot)
  oprot:writeStructBegin('ReadPost_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.post_id ~= nil then
    oprot:writeFieldBegin('post_id', TType.I64, 2)
    oprot:writeI64(self.post_id)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter154,viter155 in pairs(self.carrier) do
      oprot:writeString(kiter154)
      oprot:writeString(viter155)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

ReadPost_result = __TObject:new{
  success,
  se
}

function ReadPost_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.STRUCT then
        self.success = Post:new{}
        self.success:read(iprot)
      else
        iprot:skip(ftype)
      end
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

function ReadPost_result:write(oprot)
  oprot:writeStructBegin('ReadPost_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.STRUCT, 0)
    self.success:write(oprot)
    oprot:writeFieldEnd()
  end
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end