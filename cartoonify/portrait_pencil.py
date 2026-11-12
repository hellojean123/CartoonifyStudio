"""Local eye drawing and conservative head-envelope boundary hints, CPU only."""
import cv2
import numpy as np


def portrait_detail(prepared):
    rgb=prepared['guide'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    h,w=gray.shape;eye=np.zeros_like(gray);ink=np.zeros_like(gray);hair=np.zeros_like(gray);count=0
    # Fine eye drawing uses the un-smoothed guide, not the skin-filtered image.
    fine=cv2.bilateralFilter(np.round(gray*255).astype(np.uint8),5,18,3).astype(np.float32)/255
    for box,points in prepared.get('landmarks',[]):
        points=np.asarray(points);d=float(np.linalg.norm(points[1]-points[0]))
        if d<18 or d>max(h,w):continue
        axis=(points[1]-points[0])/d;normal=np.array([-axis[1],axis[0]])
        for center in points[:2]:
            rx=max(5,d*.23);ry=max(3,d*.11)
            x0=max(0,int(center[0]-rx*2.6));x1=min(w,int(center[0]+rx*2.6)+1)
            y0=max(0,int(center[1]-rx*1.3));y1=min(h,int(center[1]+rx*1.3)+1)
            if x1<=x0 or y1<=y0:continue
            yy,xx=np.mgrid[y0:y1,x0:x1];dx=xx-center[0];dy=yy-center[1]
            u=(dx*axis[0]+dy*axis[1])/rx;v=(dx*normal[0]+dy*normal[1])/ry
            radius=u*u+v*v;mask=np.exp(-radius*.9).astype(np.float32)
            crop=fine[y0:y1,x0:x1];selected=crop[radius<.8]
            if selected.size<12 or np.ptp(selected)<.035:continue
            lo,hi=np.quantile(selected,[.08,.92])
            if hi<.24:continue  # Insufficient exposure: keep the original drawing rather than invent contrast.
            span=max(float(hi-lo),.10)
            dark=np.clip((hi-crop)/span,0,1)
            # Source-supported dark shapes retain eyelids/iris; no invented eye geometry.
            detail=np.maximum(cv2.GaussianBlur(crop,(0,0),1.5)-crop-.008,0)
            base=.10+1.5*max(.65-float(hi),0)/.65
            global_dark=np.clip((.86-crop)/.86,0,1)
            drawing=.55*global_dark**1.2+.45*dark**1.6+np.clip(detail*4,0,.25)
            # Keep small, bright source features (including visible catchlights) white.
            drawing[(crop>=max(float(hi),.75))&(radius<.9)]=0
            mask*=float(np.clip((hi-.24)/.20,0,1))
            dest=eye[y0:y1,x0:x1];take=mask>dest
            ink[y0:y1,x0:x1][take]=drawing[take];dest[take]=mask[take];count+=1
    # Head envelopes restrict edge recovery; this is not semantic hair segmentation.
    for x,y,bw,bh in prepared.get('boxes',[]):
        x0=max(0,int(x-bw*.25));x1=min(w,int(x+bw*1.25));y0=max(0,int(y-bh*.38));y1=min(h,int(y+bh*.65))
        if x1-x0<8 or y1-y0<8:continue
        crop=gray[y0:y1,x0:x1];a=cv2.GaussianBlur(crop,(0,0),1.2)
        edges=cv2.Canny(np.round(a*255).astype(np.uint8),14,42,L2gradient=True)
        n,labels,stats,_=cv2.connectedComponentsWithStats(edges,8)
        keep=stats[:,cv2.CC_STAT_AREA]>=max(6,int(bw*.035));keep[0]=False
        lines=cv2.GaussianBlur(keep[labels].astype(np.float32),(0,0),.55)
        # Favor boundaries separating darker hair masses from lighter surroundings.
        broad=cv2.GaussianBlur(a,(0,0),max(1,bw*.014))
        gx=cv2.Sobel(broad,cv2.CV_32F,1,0)/8;gy=cv2.Sobel(broad,cv2.CV_32F,0,1)/8
        persistent=np.clip(cv2.magnitude(gx,gy)*28,0,1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        # Exclude the central face to keep new hair hints out of eyes and wrinkles.
        head=((xx-(x+bw*.5))/(bw*.85))**2+((yy-(y+bh*.12))/(bh*.68))**2
        outside_face=(1-prepared['mask'][y0:y1,x0:x1])
        zone=np.clip((1.4-head)*2,0,1)*outside_face
        hair[y0:y1,x0:x1]=np.maximum(hair[y0:y1,x0:x1],lines*persistent*zone*1.6)
    return {'eye_mask':eye,'eye_drawing':ink,'hair_boundary':np.clip(hair,0,1),'eyes_refined':count}
