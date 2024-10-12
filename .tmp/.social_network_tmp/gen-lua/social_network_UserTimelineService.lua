

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
local Post = social_network_ttypes.Post

local WriteUserTimeline_args = __TObject:new{
  req_id,
  post_id,
  user_id,
  timestamp,
  carrier
}

function WriteUserTimeline_args:read(iprot)
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
      if ftype == TType.I64 then
        self.user_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.I64 then
        self.timestamp = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 5 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype171, _vtype172, _size170 = iprot:readMapBegin()
        for _i=1,_size170 do
          local _key174 = iprot:readString()
          local _val175 = iprot:readString()
          self.carrier[_key174] = _val175
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

function WriteUserTimeline_args:write(oprot)
  oprot:writeStructBegin('WriteUserTimeline_args')
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
  if self.user_id ~= nil then
    oprot:writeFieldBegin('user_id', TType.I64, 3)
    oprot:writeI64(self.user_id)
    oprot:writeFieldEnd()
  end
  if self.timestamp ~= nil then
    oprot:writeFieldBegin('timestamp', TType.I64, 4)
    oprot:writeI64(self.timestamp)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 5)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter176,viter177 in pairs(self.carrier) do
      oprot:writeString(kiter176)
      oprot:writeString(viter177)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local WriteUserTimeline_result = __TObject:new{
  se
}

function WriteUserTimeline_result:read(iprot)
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

function WriteUserTimeline_result:write(oprot)
  oprot:writeStructBegin('WriteUserTimeline_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local ReadUserTimeline_args = __TObject:new{
  req_id,
  user_id,
  start,
  stop,
  carrier
}

function ReadUserTimeline_args:read(iprot)
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
        self.user_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.I32 then
        self.start = iprot:readI32()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.I32 then
        self.stop = iprot:readI32()
      else
        iprot:skip(ftype)
      end
    elseif fid == 5 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype179, _vtype180, _size178 = iprot:readMapBegin()
        for _i=1,_size178 do
          local _key182 = iprot:readString()
          local _val183 = iprot:readString()
          self.carrier[_key182] = _val183
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

function ReadUserTimeline_args:write(oprot)
  oprot:writeStructBegin('ReadUserTimeline_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.user_id ~= nil then
    oprot:writeFieldBegin('user_id', TType.I64, 2)
    oprot:writeI64(self.user_id)
    oprot:writeFieldEnd()
  end
  if self.start ~= nil then
    oprot:writeFieldBegin('start', TType.I32, 3)
    oprot:writeI32(self.start)
    oprot:writeFieldEnd()
  end
  if self.stop ~= nil then
    oprot:writeFieldBegin('stop', TType.I32, 4)
    oprot:writeI32(self.stop)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 5)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter184,viter185 in pairs(self.carrier) do
      oprot:writeString(kiter184)
      oprot:writeString(viter185)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local ReadUserTimeline_result = __TObject:new{
  success,
  se
}

function ReadUserTimeline_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.LIST then
        self.success = {}
        local _etype189, _size186 = iprot:readListBegin()
        for _i=1,_size186 do
          local _elem190 = Post:new{}
          _elem190:read(iprot)
          table.insert(self.success, _elem190)
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

function ReadUserTimeline_result:write(oprot)
  oprot:writeStructBegin('ReadUserTimeline_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.LIST, 0)
    oprot:writeListBegin(TType.STRUCT, #self.success)
    for _,iter191 in ipairs(self.success) do
      iter191:write(oprot)
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

local UserTimelineServiceClient = __TObject.new(__TClient, {
  __type = 'UserTimelineServiceClient'
})

function UserTimelineServiceClient:WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)
  self:send_WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)
  self:recv_WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)
end

function UserTimelineServiceClient:send_WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)
  self.oprot:writeMessageBegin('WriteUserTimeline', TMessageType.CALL, self._seqid)
  local args = WriteUserTimeline_args:new{}
  args.req_id = req_id
  args.post_id = post_id
  args.user_id = user_id
  args.timestamp = timestamp
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UserTimelineServiceClient:recv_WriteUserTimeline(req_id, post_id, user_id, timestamp, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = WriteUserTimeline_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end

function UserTimelineServiceClient:ReadUserTimeline(req_id, user_id, start, stop, carrier)
  self:send_ReadUserTimeline(req_id, user_id, start, stop, carrier)
  return self:recv_ReadUserTimeline(req_id, user_id, start, stop, carrier)
end

function UserTimelineServiceClient:send_ReadUserTimeline(req_id, user_id, start, stop, carrier)
  self.oprot:writeMessageBegin('ReadUserTimeline', TMessageType.CALL, self._seqid)
  local args = ReadUserTimeline_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.start = start
  args.stop = stop
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function UserTimelineServiceClient:recv_ReadUserTimeline(req_id, user_id, start, stop, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = ReadUserTimeline_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end
local UserTimelineServiceIface = __TObject:new{
  __type = 'UserTimelineServiceIface'
}


local UserTimelineServiceProcessor = __TObject.new(__TProcessor
, {
 __type = 'UserTimelineServiceProcessor'
})

function UserTimelineServiceProcessor:process(iprot, oprot, server_ctx)
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

function UserTimelineServiceProcessor:process_WriteUserTimeline(seqid, iprot, oprot, server_ctx)
  local args = WriteUserTimeline_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = WriteUserTimeline_result:new{}
  local status, res = pcall(self.handler.WriteUserTimeline, self.handler, args.req_id, args.post_id, args.user_id, args.timestamp, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('WriteUserTimeline', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function UserTimelineServiceProcessor:process_ReadUserTimeline(seqid, iprot, oprot, server_ctx)
  local args = ReadUserTimeline_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = ReadUserTimeline_result:new{}
  local status, res = pcall(self.handler.ReadUserTimeline, self.handler, args.req_id, args.user_id, args.start, args.stop, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('ReadUserTimeline', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

return UserTimelineServiceClient