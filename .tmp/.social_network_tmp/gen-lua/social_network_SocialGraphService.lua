
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


local GetFollowers_args = __TObject:new{
  req_id,
  user_id,
  carrier
}

function GetFollowers_args:read(iprot)
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
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype201, _vtype202, _size200 = iprot:readMapBegin()
        for _i=1,_size200 do
          local _key204 = iprot:readString()
          local _val205 = iprot:readString()
          self.carrier[_key204] = _val205
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

function GetFollowers_args:write(oprot)
  oprot:writeStructBegin('GetFollowers_args')
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
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter206,viter207 in pairs(self.carrier) do
      oprot:writeString(kiter206)
      oprot:writeString(viter207)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local GetFollowers_result = __TObject:new{
  success,
  se
}

function GetFollowers_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.LIST then
        self.success = {}
        local _etype211, _size208 = iprot:readListBegin()
        for _i=1,_size208 do
          local _elem212 = iprot:readI64()
          table.insert(self.success, _elem212)
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

function GetFollowers_result:write(oprot)
  oprot:writeStructBegin('GetFollowers_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.LIST, 0)
    oprot:writeListBegin(TType.I64, #self.success)
    for _,iter213 in ipairs(self.success) do
      oprot:writeI64(iter213)
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

local GetFollowees_args = __TObject:new{
  req_id,
  user_id,
  carrier
}

function GetFollowees_args:read(iprot)
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
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype215, _vtype216, _size214 = iprot:readMapBegin()
        for _i=1,_size214 do
          local _key218 = iprot:readString()
          local _val219 = iprot:readString()
          self.carrier[_key218] = _val219
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

function GetFollowees_args:write(oprot)
  oprot:writeStructBegin('GetFollowees_args')
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
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter220,viter221 in pairs(self.carrier) do
      oprot:writeString(kiter220)
      oprot:writeString(viter221)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local GetFollowees_result = __TObject:new{
  success,
  se
}

function GetFollowees_result:read(iprot)
  iprot:readStructBegin()
  while true do
    local fname, ftype, fid = iprot:readFieldBegin()
    if ftype == TType.STOP then
      break
    elseif fid == 0 then
      if ftype == TType.LIST then
        self.success = {}
        local _etype225, _size222 = iprot:readListBegin()
        for _i=1,_size222 do
          local _elem226 = iprot:readI64()
          table.insert(self.success, _elem226)
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

function GetFollowees_result:write(oprot)
  oprot:writeStructBegin('GetFollowees_result')
  if self.success ~= nil then
    oprot:writeFieldBegin('success', TType.LIST, 0)
    oprot:writeListBegin(TType.I64, #self.success)
    for _,iter227 in ipairs(self.success) do
      oprot:writeI64(iter227)
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

local Follow_args = __TObject:new{
  req_id,
  user_id,
  followee_id,
  carrier
}

function Follow_args:read(iprot)
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
      if ftype == TType.I64 then
        self.followee_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype229, _vtype230, _size228 = iprot:readMapBegin()
        for _i=1,_size228 do
          local _key232 = iprot:readString()
          local _val233 = iprot:readString()
          self.carrier[_key232] = _val233
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

function Follow_args:write(oprot)
  oprot:writeStructBegin('Follow_args')
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
  if self.followee_id ~= nil then
    oprot:writeFieldBegin('followee_id', TType.I64, 3)
    oprot:writeI64(self.followee_id)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter234,viter235 in pairs(self.carrier) do
      oprot:writeString(kiter234)
      oprot:writeString(viter235)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local Follow_result = __TObject:new{
  se
}

function Follow_result:read(iprot)
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

function Follow_result:write(oprot)
  oprot:writeStructBegin('Follow_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local Unfollow_args = __TObject:new{
  req_id,
  user_id,
  followee_id,
  carrier
}

function Unfollow_args:read(iprot)
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
      if ftype == TType.I64 then
        self.followee_id = iprot:readI64()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype237, _vtype238, _size236 = iprot:readMapBegin()
        for _i=1,_size236 do
          local _key240 = iprot:readString()
          local _val241 = iprot:readString()
          self.carrier[_key240] = _val241
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

function Unfollow_args:write(oprot)
  oprot:writeStructBegin('Unfollow_args')
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
  if self.followee_id ~= nil then
    oprot:writeFieldBegin('followee_id', TType.I64, 3)
    oprot:writeI64(self.followee_id)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter242,viter243 in pairs(self.carrier) do
      oprot:writeString(kiter242)
      oprot:writeString(viter243)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local Unfollow_result = __TObject:new{
  se
}

function Unfollow_result:read(iprot)
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

function Unfollow_result:write(oprot)
  oprot:writeStructBegin('Unfollow_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local FollowWithUsername_args = __TObject:new{
  req_id,
  user_usernmae,
  followee_username,
  carrier
}

function FollowWithUsername_args:read(iprot)
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
        self.user_usernmae = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.STRING then
        self.followee_username = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype245, _vtype246, _size244 = iprot:readMapBegin()
        for _i=1,_size244 do
          local _key248 = iprot:readString()
          local _val249 = iprot:readString()
          self.carrier[_key248] = _val249
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

function FollowWithUsername_args:write(oprot)
  oprot:writeStructBegin('FollowWithUsername_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.user_usernmae ~= nil then
    oprot:writeFieldBegin('user_usernmae', TType.STRING, 2)
    oprot:writeString(self.user_usernmae)
    oprot:writeFieldEnd()
  end
  if self.followee_username ~= nil then
    oprot:writeFieldBegin('followee_username', TType.STRING, 3)
    oprot:writeString(self.followee_username)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter250,viter251 in pairs(self.carrier) do
      oprot:writeString(kiter250)
      oprot:writeString(viter251)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local FollowWithUsername_result = __TObject:new{
  se
}

function FollowWithUsername_result:read(iprot)
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

function FollowWithUsername_result:write(oprot)
  oprot:writeStructBegin('FollowWithUsername_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local UnfollowWithUsername_args = __TObject:new{
  req_id,
  user_usernmae,
  followee_username,
  carrier
}

function UnfollowWithUsername_args:read(iprot)
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
        self.user_usernmae = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 3 then
      if ftype == TType.STRING then
        self.followee_username = iprot:readString()
      else
        iprot:skip(ftype)
      end
    elseif fid == 4 then
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype253, _vtype254, _size252 = iprot:readMapBegin()
        for _i=1,_size252 do
          local _key256 = iprot:readString()
          local _val257 = iprot:readString()
          self.carrier[_key256] = _val257
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

function UnfollowWithUsername_args:write(oprot)
  oprot:writeStructBegin('UnfollowWithUsername_args')
  if self.req_id ~= nil then
    oprot:writeFieldBegin('req_id', TType.I64, 1)
    oprot:writeI64(self.req_id)
    oprot:writeFieldEnd()
  end
  if self.user_usernmae ~= nil then
    oprot:writeFieldBegin('user_usernmae', TType.STRING, 2)
    oprot:writeString(self.user_usernmae)
    oprot:writeFieldEnd()
  end
  if self.followee_username ~= nil then
    oprot:writeFieldBegin('followee_username', TType.STRING, 3)
    oprot:writeString(self.followee_username)
    oprot:writeFieldEnd()
  end
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 4)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter258,viter259 in pairs(self.carrier) do
      oprot:writeString(kiter258)
      oprot:writeString(viter259)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local UnfollowWithUsername_result = __TObject:new{
  se
}

function UnfollowWithUsername_result:read(iprot)
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

function UnfollowWithUsername_result:write(oprot)
  oprot:writeStructBegin('UnfollowWithUsername_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local InsertUser_args = __TObject:new{
  req_id,
  user_id,
  carrier
}

function InsertUser_args:read(iprot)
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
      if ftype == TType.MAP then
        self.carrier = {}
        local _ktype261, _vtype262, _size260 = iprot:readMapBegin()
        for _i=1,_size260 do
          local _key264 = iprot:readString()
          local _val265 = iprot:readString()
          self.carrier[_key264] = _val265
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

function InsertUser_args:write(oprot)
  oprot:writeStructBegin('InsertUser_args')
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
  if self.carrier ~= nil then
    oprot:writeFieldBegin('carrier', TType.MAP, 3)
    oprot:writeMapBegin(TType.STRING, TType.STRING, ttable_size(self.carrier))
    for kiter266,viter267 in pairs(self.carrier) do
      oprot:writeString(kiter266)
      oprot:writeString(viter267)
    end
    oprot:writeMapEnd()
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local InsertUser_result = __TObject:new{
  se
}

function InsertUser_result:read(iprot)
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

function InsertUser_result:write(oprot)
  oprot:writeStructBegin('InsertUser_result')
  if self.se ~= nil then
    oprot:writeFieldBegin('se', TType.STRUCT, 1)
    self.se:write(oprot)
    oprot:writeFieldEnd()
  end
  oprot:writeFieldStop()
  oprot:writeStructEnd()
end

local SocialGraphServiceClient = __TObject.new(__TClient, {
  __type = 'SocialGraphServiceClient'
})

function SocialGraphServiceClient:GetFollowers(req_id, user_id, carrier)
  self:send_GetFollowers(req_id, user_id, carrier)
  return self:recv_GetFollowers(req_id, user_id, carrier)
end

function SocialGraphServiceClient:send_GetFollowers(req_id, user_id, carrier)
  self.oprot:writeMessageBegin('GetFollowers', TMessageType.CALL, self._seqid)
  local args = GetFollowers_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_GetFollowers(req_id, user_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = GetFollowers_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end

function SocialGraphServiceClient:GetFollowees(req_id, user_id, carrier)
  self:send_GetFollowees(req_id, user_id, carrier)
  return self:recv_GetFollowees(req_id, user_id, carrier)
end

function SocialGraphServiceClient:send_GetFollowees(req_id, user_id, carrier)
  self.oprot:writeMessageBegin('GetFollowees', TMessageType.CALL, self._seqid)
  local args = GetFollowees_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_GetFollowees(req_id, user_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = GetFollowees_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.success ~= nil then
    return result.success
  elseif result.se then
    error(result.se)
  end
  error(TApplicationException:new{errorCode = TApplicationException.MISSING_RESULT})
end

function SocialGraphServiceClient:Follow(req_id, user_id, followee_id, carrier)
  self:send_Follow(req_id, user_id, followee_id, carrier)
  self:recv_Follow(req_id, user_id, followee_id, carrier)
end

function SocialGraphServiceClient:send_Follow(req_id, user_id, followee_id, carrier)
  self.oprot:writeMessageBegin('Follow', TMessageType.CALL, self._seqid)
  local args = Follow_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.followee_id = followee_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_Follow(req_id, user_id, followee_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = Follow_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end

function SocialGraphServiceClient:Unfollow(req_id, user_id, followee_id, carrier)
  self:send_Unfollow(req_id, user_id, followee_id, carrier)
  self:recv_Unfollow(req_id, user_id, followee_id, carrier)
end

function SocialGraphServiceClient:send_Unfollow(req_id, user_id, followee_id, carrier)
  self.oprot:writeMessageBegin('Unfollow', TMessageType.CALL, self._seqid)
  local args = Unfollow_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.followee_id = followee_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_Unfollow(req_id, user_id, followee_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = Unfollow_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end

function SocialGraphServiceClient:FollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self:send_FollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self:recv_FollowWithUsername(req_id, user_usernmae, followee_username, carrier)
end

function SocialGraphServiceClient:send_FollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self.oprot:writeMessageBegin('FollowWithUsername', TMessageType.CALL, self._seqid)
  local args = FollowWithUsername_args:new{}
  args.req_id = req_id
  args.user_usernmae = user_usernmae
  args.followee_username = followee_username
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_FollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = FollowWithUsername_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end

function SocialGraphServiceClient:UnfollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self:send_UnfollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self:recv_UnfollowWithUsername(req_id, user_usernmae, followee_username, carrier)
end

function SocialGraphServiceClient:send_UnfollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  self.oprot:writeMessageBegin('UnfollowWithUsername', TMessageType.CALL, self._seqid)
  local args = UnfollowWithUsername_args:new{}
  args.req_id = req_id
  args.user_usernmae = user_usernmae
  args.followee_username = followee_username
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_UnfollowWithUsername(req_id, user_usernmae, followee_username, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = UnfollowWithUsername_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end

function SocialGraphServiceClient:InsertUser(req_id, user_id, carrier)
  self:send_InsertUser(req_id, user_id, carrier)
  self:recv_InsertUser(req_id, user_id, carrier)
end

function SocialGraphServiceClient:send_InsertUser(req_id, user_id, carrier)
  self.oprot:writeMessageBegin('InsertUser', TMessageType.CALL, self._seqid)
  local args = InsertUser_args:new{}
  args.req_id = req_id
  args.user_id = user_id
  args.carrier = carrier
  args:write(self.oprot)
  self.oprot:writeMessageEnd()
  self.oprot.trans:flush()
end

function SocialGraphServiceClient:recv_InsertUser(req_id, user_id, carrier)
  local fname, mtype, rseqid = self.iprot:readMessageBegin()
  if mtype == TMessageType.EXCEPTION then
    local x = TApplicationException:new{}
    x:read(self.iprot)
    self.iprot:readMessageEnd()
    error(x)
  end
  local result = InsertUser_result:new{}
  result:read(self.iprot)
  self.iprot:readMessageEnd()
  if result.se then
    error(result.se)
  end
end
local SocialGraphServiceIface = __TObject:new{
  __type = 'SocialGraphServiceIface'
}


local SocialGraphServiceProcessor = __TObject.new(__TProcessor
, {
      __type = 'SocialGraphServiceProcessor'
    })

function SocialGraphServiceProcessor:process(iprot, oprot, server_ctx)
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

function SocialGraphServiceProcessor:process_GetFollowers(seqid, iprot, oprot, server_ctx)
  local args = GetFollowers_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = GetFollowers_result:new{}
  local status, res = pcall(self.handler.GetFollowers, self.handler, args.req_id, args.user_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('GetFollowers', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_GetFollowees(seqid, iprot, oprot, server_ctx)
  local args = GetFollowees_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = GetFollowees_result:new{}
  local status, res = pcall(self.handler.GetFollowees, self.handler, args.req_id, args.user_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('GetFollowees', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_Follow(seqid, iprot, oprot, server_ctx)
  local args = Follow_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = Follow_result:new{}
  local status, res = pcall(self.handler.Follow, self.handler, args.req_id, args.user_id, args.followee_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('Follow', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_Unfollow(seqid, iprot, oprot, server_ctx)
  local args = Unfollow_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = Unfollow_result:new{}
  local status, res = pcall(self.handler.Unfollow, self.handler, args.req_id, args.user_id, args.followee_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('Unfollow', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_FollowWithUsername(seqid, iprot, oprot, server_ctx)
  local args = FollowWithUsername_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = FollowWithUsername_result:new{}
  local status, res = pcall(self.handler.FollowWithUsername, self.handler, args.req_id, args.user_usernmae, args.followee_username, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('FollowWithUsername', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_UnfollowWithUsername(seqid, iprot, oprot, server_ctx)
  local args = UnfollowWithUsername_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = UnfollowWithUsername_result:new{}
  local status, res = pcall(self.handler.UnfollowWithUsername, self.handler, args.req_id, args.user_usernmae, args.followee_username, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('UnfollowWithUsername', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

function SocialGraphServiceProcessor:process_InsertUser(seqid, iprot, oprot, server_ctx)
  local args = InsertUser_args:new{}
  local reply_type = TMessageType.REPLY
  args:read(iprot)
  iprot:readMessageEnd()
  local result = InsertUser_result:new{}
  local status, res = pcall(self.handler.InsertUser, self.handler, args.req_id, args.user_id, args.carrier)
  if not status then
    reply_type = TMessageType.EXCEPTION
    result = TApplicationException:new{message = res}
  elseif ttype(res) == 'ServiceException' then
    result.se = res
  else
    result.success = res
  end
  oprot:writeMessageBegin('InsertUser', reply_type, seqid)
  result:write(oprot)
  oprot:writeMessageEnd()
  oprot.trans:flush()
end

return SocialGraphServiceClient