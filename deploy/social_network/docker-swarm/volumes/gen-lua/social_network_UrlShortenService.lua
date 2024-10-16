require 'Thrift'
require 'social_network_ttypes'

UrlShortenServiceClient = __TObject.new(__TClient, {
  __type = 'UrlShortenServiceClient'
})

function UrlShortenServiceClient:UploadUrls(req_id, urls, carrier)
  self:send_UploadUrls(req_id, urls, carrier)
  return self:recv_UploadUrls(req_id, urls, carrier)
end

function UrlShortenServiceClient:send_UploadUrls(req_id, urls, carrier)
  self.oprot:writeMessageBegin('UploadUrls', TMessageType.CALL, self._seqid)
  local args = UploadUrls_args:new{}
  args.req_id = req_id
  args.urls = urls
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UrlShortenServiceClient:recv_UploadUrls(req_id, urls, carrier)
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
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end

function UrlShortenServiceClient:GetExtendedUrls(req_id, shortened_urls, carrier)
  self:send_GetExtendedUrls(req_id, shortened_urls, carrier)
  return self:recv_GetExtendedUrls(req_id, shortened_urls, carrier)
end

function UrlShortenServiceClient:send_GetExtendedUrls(req_id, shortened_urls, carrier)
  self.oprot:writeMessageBegin('GetExtendedUrls', TMessageType.CALL, self._seqid)
  local args = GetExtendedUrls_args:new{}
  args.req_id = req_id
  args.shortened_urls = shortened_urls
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UrlShortenServiceClient:recv_GetExtendedUrls(req_id, shortened_urls, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = GetExtendedUrls_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end
UrlShortenServiceIface = __TObject:new{
  __type = 'UrlShortenServiceIface'
}


UrlShortenServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'UrlShortenServiceProcessor'
})

function UrlShortenServiceProcessor:process(iprot, oprot, server_ctx)
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

function UrlShortenServiceProcessor:process_UploadUrls(seqid, iprot, oprot, server_ctx)
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

function UrlShortenServiceProcessor:process_GetExtendedUrls(seqid, iprot, oprot, server_ctx)
  local args = GetExtendedUrls_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = GetExtendedUrls_result:new{}
  local status, res = pcall(self.handler.GetExtendedUrls, self.handler, args.req_id, args.shortened_urls, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('GetExtendedUrls', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
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
        local _etype253, _size250 = iprot:readListBegin()
        for _i=1,_size250 do
          local _elem254 = iprot:readString()
          table.insert(self.urls, _elem254)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype256, _vtype257, _size255 = iprot:readMapBegin() 
        for _i=1,_size255 do
          local _key259 = iprot:readString()
          local _val260 = iprot:readString()
          self.carrier[_key259] = _val260
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
    oprot:writeListBegin(TType.STRING, #self.urls)
    for _,iter261 in ipairs(self.urls) do
      oprot:writeString(iter261)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter262,viter263 in pairs(self.carrier) do
      oprot:writeString(kiter262)
      oprot:writeString(viter263)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

UploadUrls_result = __TObject:new{
  success,
  se
}

function UploadUrls_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.LIST then
        self.success = {}
        local _etype267, _size264 = iprot:readListBegin()
        for _i=1,_size264 do
          local _elem268 = iprot:readString()
          table.insert(self.success, _elem268)
        end
        iprot:readListEnd()
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

function UploadUrls_result:write(oprot)
  oprot:writeStructBegin('UploadUrls_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.LIST, 0)
    oprot:writeListBegin(TType.STRING, #self.success)
    for _,iter269 in ipairs(self.success) do
      oprot:writeString(iter269)
    end
    oprot:writeListEnd()
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

GetExtendedUrls_args = __TObject:new{
  req_id,
  shortened_urls,
  carrier
}

function GetExtendedUrls_args:read(iprot)
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
        self.shortened_urls = {}
        local _etype273, _size270 = iprot:readListBegin()
        for _i=1,_size270 do
          local _elem274 = iprot:readString()
          table.insert(self.shortened_urls, _elem274)
        end
        iprot:readListEnd()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype276, _vtype277, _size275 = iprot:readMapBegin() 
        for _i=1,_size275 do
          local _key279 = iprot:readString()
          local _val280 = iprot:readString()
          self.carrier[_key279] = _val280
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

function GetExtendedUrls_args:write(oprot)
  oprot:writeStructBegin('GetExtendedUrls_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.shortened_urls ~= nil then
    oprot:writeFieldBegin('shortened_urls', TType.LIST, 2)
    oprot:writeListBegin(TType.STRING, #self.shortened_urls)
    for _,iter281 in ipairs(self.shortened_urls) do
      oprot:writeString(iter281)
    end
    oprot:writeListEnd()
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter282,viter283 in pairs(self.carrier) do
      oprot:writeString(kiter282)
      oprot:writeString(viter283)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

GetExtendedUrls_result = __TObject:new{
  success,
  se
}

function GetExtendedUrls_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.LIST then
        self.success = {}
        local _etype287, _size284 = iprot:readListBegin()
        for _i=1,_size284 do
          local _elem288 = iprot:readString()
          table.insert(self.success, _elem288)
        end
        iprot:readListEnd()
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

function GetExtendedUrls_result:write(oprot)
  oprot:writeStructBegin('GetExtendedUrls_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.LIST, 0)
    oprot:writeListBegin(TType.STRING, #self.success)
    for _,iter289 in ipairs(self.success) do
      oprot:writeString(iter289)
    end
    oprot:writeListEnd()
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