require 'Thrift'
require 'social_network_ttypes'

ComposePostServiceClient = __TObject.new(__TClient, {
  __type = 'ComposePostServiceClient'
})

function ComposePostServiceClient:UploadText(req_id, text, carrier)
  self:send_UploadText(req_id, text, carrier)
  self:recv_UploadText(req_id, text, carrier)
end

function ComposePostServiceClient:send_UploadText(req_id, text, carrier)
  self.oprot:writeMessageBegin('UploadText', TMessageType.CALL, self._seqid)
  local args = UploadText_args:new{}
  args.req_id = req_id
  args.text = text
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadText(req_id, text, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadText_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end

function ComposePostServiceClient:UploadMedia(req_id, media, carrier)
  self:send_UploadMedia(req_id, media, carrier)
  self:recv_UploadMedia(req_id, media, carrier)
end

function ComposePostServiceClient:send_UploadMedia(req_id, media, carrier)
  self.oprot:writeMessageBegin('UploadMedia', TMessageType.CALL, self._seqid)
  local args = UploadMedia_args:new{}
  args.req_id = req_id
  args.media = media
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadMedia(req_id, media, carrier)
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

function ComposePostServiceClient:UploadUniqueId(req_id, post_id, post_type, carrier)
  self:send_UploadUniqueId(req_id, post_id, post_type, carrier)
  self:recv_UploadUniqueId(req_id, post_id, post_type, carrier)
end

function ComposePostServiceClient:send_UploadUniqueId(req_id, post_id, post_type, carrier)
  self.oprot:writeMessageBegin('UploadUniqueId', TMessageType.CALL, self._seqid)
  local args = UploadUniqueId_args:new{}
  args.req_id = req_id
  args.post_id = post_id
  args.post_type = post_type
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadUniqueId(req_id, post_id, post_type, carrier)
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
end

function ComposePostServiceClient:UploadCreator(req_id, creator, carrier)
  self:send_UploadCreator(req_id, creator, carrier)
  self:recv_UploadCreator(req_id, creator, carrier)
end

function ComposePostServiceClient:send_UploadCreator(req_id, creator, carrier)
  self.oprot:writeMessageBegin('UploadCreator', TMessageType.CALL, self._seqid)
  local args = UploadCreator_args:new{}
  args.req_id = req_id
  args.creator = creator
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadCreator(req_id, creator, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadCreator_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end

function ComposePostServiceClient:UploadUrls(req_id, urls, carrier)
  self:send_UploadUrls(req_id, urls, carrier)
  self:recv_UploadUrls(req_id, urls, carrier)
end

function ComposePostServiceClient:send_UploadUrls(req_id, urls, carrier)
  self.oprot:writeMessageBegin('UploadUrls', TMessageType.CALL, self._seqid)
  local args = UploadUrls_args:new{}
  args.req_id = req_id
  args.urls = urls
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadUrls(req_id, urls, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UploadUrls_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
end

function ComposePostServiceClient:UploadUserMentions(req_id, user_mentions, carrier)
  self:send_UploadUserMentions(req_id, user_mentions, carrier)
  self:recv_UploadUserMentions(req_id, user_mentions, carrier)
end

function ComposePostServiceClient:send_UploadUserMentions(req_id, user_mentions, carrier)
  self.oprot:writeMessageBegin('UploadUserMentions', TMessageType.CALL, self._seqid)
  local args = UploadUserMentions_args:new{}
  args.req_id = req_id
  args.user_mentions = user_mentions
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function ComposePostServiceClient:recv_UploadUserMentions(req_id, user_mentions, carrier)
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
ComposePostServiceIface = __TObject:new{
  __type = 'ComposePostServiceIface'
}


ComposePostServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'ComposePostServiceProcessor'
})

function ComposePostServiceProcessor:process(iprot, oprot, server_ctx)
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

function ComposePostServiceProcessor:process_UploadText(seqid, iprot, oprot, server_ctx)
  local args = UploadText_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadText_result:new{}
  local status, res = pcall(self.handler.UploadText, self.handler, args.req_id, args.text, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadText', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function ComposePostServiceProcessor:process_UploadMedia(seqid, iprot, oprot, server_ctx)
  local args = UploadMedia_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadMedia_result:new{}
  local status, res = pcall(self.handler.UploadMedia, self.handler, args.req_id, args.media, args.carrier)
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

function ComposePostServiceProcessor:process_UploadUniqueId(seqid, iprot, oprot, server_ctx)
  local args = UploadUniqueId_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadUniqueId_result:new{}
  local status, res = pcall(self.handler.UploadUniqueId, self.handler, args.req_id, args.post_id, args.post_type, args.carrier)
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

function ComposePostServiceProcessor:process_UploadCreator(seqid, iprot, oprot, server_ctx)
  local args = UploadCreator_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadCreator_result:new{}
  local status, res = pcall(self.handler.UploadCreator, self.handler, args.req_id, args.creator, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadCreator', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function ComposePostServiceProcessor:process_UploadUrls(seqid, iprot, oprot, server_ctx)
  local args = UploadUrls_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadUrls_result:new{}
  local status, res = pcall(self.handler.UploadUrls, self.handler, args.req_id, args.urls, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UploadUrls', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function ComposePostServiceProcessor:process_UploadUserMentions(seqid, iprot, oprot, server_ctx)
  local args = UploadUserMentions_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UploadUserMentions_result:new{}
  local status, res = pcall(self.handler.UploadUserMentions, self.handler, args.req_id, args.user_mentions, args.carrier)
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

UploadText_args = __TObject:new{
  req_id,
  text,
  carrier
}

function UploadText_args:read(iprot)
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
      if ftype == TType.STRING then
        self.text = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype75, _vtype76, _size74 = iprot:readMapBegin() 
        for _i=1,_size74 do
          local _key78 = iprot:readString()
          local _val79 = iprot:readString()
          self.carrier[_key78] = _val79
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

function UploadText_args:write(oprot)
  oprot:writeStructBegin('UploadText_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.text ~= nil then
    oprot:writeFieldBegin('text', TType.STRING, 2)
    oprot:writeString(self.text)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter80,viter81 in pairs(self.carrier) do
      oprot:writeString(kiter80)
      oprot:writeString(viter81)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadText_result = __TObject:new{
  se
}

function UploadText_result:read(iprot)
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

function UploadText_result:write(oprot)
  oprot:writeStructBegin('UploadText_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadMedia_args = __TObject:new{
  req_id,
  media,
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
        self.media = {}
        local _etype85, _size82 = iprot:readListBegin()
        for _i=1,_size82 do
          local _elem86 = Media:new{}
          _elem86:read(iprot)
          table.insert(self.media, _elem86)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype88, _vtype89, _size87 = iprot:readMapBegin() 
        for _i=1,_size87 do
          local _key91 = iprot:readString()
          local _val92 = iprot:readString()
          self.carrier[_key91] = _val92
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
  if self.media ~= nil then
    oprot:writeFieldBegin('media', TType.LIST, 2)
    oprot:writeListBegin(TType.STRUCT, #self.media)
    for _,iter93 in ipairs(self.media) do
      iter93:write(oprot)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter94,viter95 in pairs(self.carrier) do
      oprot:writeString(kiter94)
      oprot:writeString(viter95)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadMedia_result = __TObject:new{
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

UploadUniqueId_args = __TObject:new{
  req_id,
  post_id,
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
      if ftype == TType.I64 then
        self.post_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.I32 then
        self.post_type = iprot:readI32()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype97, _vtype98, _size96 = iprot:readMapBegin() 
        for _i=1,_size96 do
          local _key100 = iprot:readString()
          local _val101 = iprot:readString()
          self.carrier[_key100] = _val101
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
  if self.post_id ~= nil then
    oprot:writeFieldBegin('post_id', TType.I64, 2)
    oprot:writeI64(self.post_id)
    oprot:writeFieldEnd()
  end
  if self.post_type ~= nil then
    oprot:writeFieldBegin('post_type', TType.I32, 3)
    oprot:writeI32(self.post_type)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter102,viter103 in pairs(self.carrier) do
      oprot:writeString(kiter102)
      oprot:writeString(viter103)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUniqueId_result = __TObject:new{
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

UploadCreator_args = __TObject:new{
  req_id,
  creator,
  carrier
}

function UploadCreator_args:read(iprot)
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
        self.creator = Creator:new{}
        self.creator:read(iprot)
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype105, _vtype106, _size104 = iprot:readMapBegin() 
        for _i=1,_size104 do
          local _key108 = iprot:readString()
          local _val109 = iprot:readString()
          self.carrier[_key108] = _val109
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

function UploadCreator_args:write(oprot)
  oprot:writeStructBegin('UploadCreator_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.creator ~= nil then
    oprot:writeFieldBegin('creator', TType.STRUCT, 2)
    self.creator:write(oprot)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter110,viter111 in pairs(self.carrier) do
      oprot:writeString(kiter110)
      oprot:writeString(viter111)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadCreator_result = __TObject:new{
  se
}

function UploadCreator_result:read(iprot)
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

function UploadCreator_result:write(oprot)
  oprot:writeStructBegin('UploadCreator_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUrls_args = __TObject:new{
  req_id,
  urls,
  carrier
}

function UploadUrls_args:read(iprot)
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
        self.urls = {}
        local _etype115, _size112 = iprot:readListBegin()
        for _i=1,_size112 do
          local _elem116 = Url:new{}
          _elem116:read(iprot)
          table.insert(self.urls, _elem116)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype118, _vtype119, _size117 = iprot:readMapBegin() 
        for _i=1,_size117 do
          local _key121 = iprot:readString()
          local _val122 = iprot:readString()
          self.carrier[_key121] = _val122
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

function UploadUrls_args:write(oprot)
  oprot:writeStructBegin('UploadUrls_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.urls ~= nil then
    oprot:writeFieldBegin('urls', TType.LIST, 2)
    oprot:writeListBegin(TType.STRUCT, #self.urls)
    for _,iter123 in ipairs(self.urls) do
      iter123:write(oprot)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter124,viter125 in pairs(self.carrier) do
      oprot:writeString(kiter124)
      oprot:writeString(viter125)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUrls_result = __TObject:new{
  se
}

function UploadUrls_result:read(iprot)
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

function UploadUrls_result:write(oprot)
  oprot:writeStructBegin('UploadUrls_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUserMentions_args = __TObject:new{
  req_id,
  user_mentions,
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
        self.user_mentions = {}
        local _etype129, _size126 = iprot:readListBegin()
        for _i=1,_size126 do
          local _elem130 = UserMention:new{}
          _elem130:read(iprot)
          table.insert(self.user_mentions, _elem130)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype132, _vtype133, _size131 = iprot:readMapBegin() 
        for _i=1,_size131 do
          local _key135 = iprot:readString()
          local _val136 = iprot:readString()
          self.carrier[_key135] = _val136
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
  if self.user_mentions ~= nil then
    oprot:writeFieldBegin('user_mentions', TType.LIST, 2)
    oprot:writeListBegin(TType.STRUCT, #self.user_mentions)
    for _,iter137 in ipairs(self.user_mentions) do
      iter137:write(oprot)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter138,viter139 in pairs(self.carrier) do
      oprot:writeString(kiter138)
      oprot:writeString(viter139)
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