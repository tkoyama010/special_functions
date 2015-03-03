#  !/usr/bin/python -t
#   -*- coding: utf-8 -*-
import numpy as np
from scipy import special
R = 18
@np.vectorize
def sv(n,r):
  '''
  Struve 関数
  '''
  if(np.abs(r)<R):
    k = np.arange(0,100,1)
    return np.sum((-1)**k*(r/2.0)**(2*k+n+1)/special.gamma(k+(3.0/2.0))/special.gamma(n+k+(3.0/2.0)))
  else:
    return special.yv(n,r)+F(n,r)
@np.vectorize
def dr1sv(n,r):
  '''
  Struve関数のrに対する一階微分
  '''
  if(n == 0):
    return (2.0/np.pi)-sv(1,r)
  else:
    return (sv(n-1,r)-sv(n+1,r)+(r/2.)**n/np.sqrt(np.pi)/special.gamma(n+(3./2.)))/2.
@np.vectorize
def dr1yv(n,r):
  '''
  ノイマン関数のrに対する一階微分
  '''
  return (special.yv(n-1,r)-special.yv(n+1,r))/2.0
@np.vectorize
def dr1hankel1(n,r):
  return (special.hankel1(n-1,r)-special.hankel1(n+1,r))/2.0
@np.vectorize
def dr1hankel2(n,r):
  return (special.hankel2(n-1,r)-special.hankel2(n+1,r))/2.0
@np.vectorize
def f(n,r):
  p = R
  k = np.arange(0,p-1+1,1)
  return 1.0/np.pi*np.sum(special.gamma(k+1.0/2.0)*(r/2.0)**(-2.0*k+n-1)/special.gamma(n-k+1.0/2.0))
@np.vectorize
def dr1f(n,r):
  p = R
  k = np.arange(0,p-1+1,1)
  return 1.0/np.pi*np.sum(special.gamma(k+1.0/2.0)*(-2.0*k+n-1)/2.0*(r/2.0)**(-2.0*k+n-2)/special.gamma(n-k+1.0/2.0))
@np.vectorize
def F(n,r):
  if(np.abs(r)<R):
    return sv(n,r)-special.yv(n,r)
  else:
    if(np.angle(r)<np.pi/2.0):
      return f(n,r)
    elif(np.angle(r)<np.pi):
      return -(-1.0)**n*(f(n,-r)+2.0*np.complex128(0+1j)*special.hankel2(n,-r))
    elif(np.angle(r)<np.pi*3.0/2.0):
      return -(-1.0)**n*(f(n,-r)-2.0*np.complex128(0+1j)*special.hankel1(n,-r))
    elif(np.angle(r)<=2.0*np.pi):
      return f(n,r)
@np.vectorize
def dr1F(n,r):
  if(np.abs(r)<R):
    return dr1sv(n,r)-dr1yv(n,r)
  else:
    if(np.angle(r)<np.pi/2.0):
      return dr1f(n,r)
    elif(np.angle(r)<np.pi):
      return -(-1.0)**n*(-dr1f(n,-r)-2.0*np.complex128(0+1j)*dr1hankel2(n,-r))
    elif(np.angle(r)<np.pi*3.0/2.0):
      return -(-1.0)**n*(-dr1f(n,-r)+2.0*np.complex128(0+1j)*dr1hankel1(n,-r))
    elif(np.angle(r)<=2.0*np.pi):
      return dr1f(n,r)
def I01(kn,r):
  return  1./2.*(1./r/kn+np.pi/2.*F(0,-kn*r))
def dr1I01(kn,r):
  return  1./2.*(-1./r**2/kn-kn*np.pi/2.*dr1F(0,-kn*r))
def I02(kn,r):
  return  1./2.*(1.-np.pi/2.*F(1,-kn*r))
def dr1I02(kn,r):
  return  1./2.*(kn*np.pi/2.*dr1F(1,-kn*r))
def I03(kn,r):
  return  1./2./kn*(1.-1./kn/r-np.pi/2.*F(1,-kn*r))
def dr1I03(kn,r):
  return  I01(kn,r)-1./r*I03(kn,r)
def dr2I03(kn,r):
  dr2I03 = dr1I01(kn,r)-(-1./r**2*I03(kn,r)+1./r*dr1I03(kn,r))
  return  dr2I03
def dr1I03r(kn,r):
  dr1I03r = dr1I03(kn,r)/r-I03(kn,r)/r**2
  return dr1I03r
