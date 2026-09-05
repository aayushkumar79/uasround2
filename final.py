import numpy as np
import cv2
import math
import heapq

img=cv2.imread("D:/ML/IMG-20260831-WA0026.jpg")
output=img.copy()
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#retval,otsu=cv2.threshold(gray,1,255,cv2.THRESH_OTSU)

#Creating individual masks for violet and black
lower_violet=np.array([120,50,50])
upper_violet=np.array([140,255,255])
mask_violet=cv2.inRange(hsv,lower_violet,upper_violet)
lower_black=np.array([0,0,0])
upper_black=np.array([179,255,50])
mask_black=cv2.inRange(hsv,lower_black,upper_black)

#Creating masks for different colors
lower_red1=np.array([0,100,100])
upper_red1=np.array([10,255,255])
lower_red2=np.array([170,120,70])
upper_red2=np.array([180,255,255])
mask_red=cv2.bitwise_or(cv2.inRange(hsv,lower_red1,upper_red1),cv2.inRange(hsv,lower_red2,upper_red2))
lower_purple=np.array([140,50,100])
upper_purple=np.array([170,255,255])
mask_purple=cv2.inRange(hsv,lower_purple,upper_purple)
lower_yellow=np.array([20,100,100])
upper_yellow=np.array([35,255,255])
mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
lower_white=np.array([0,0,200])
upper_white=np.array([180,50,255])
mask_white=cv2.inRange(hsv,lower_white,upper_white)
lower_orange=np.array([11,100,100])
upper_orange=np.array([19,255,255])
mask_orange=cv2.inRange(hsv,lower_orange,upper_orange)

#Merging all the colors masks together
masks=[("red",mask_red),("yellow",mask_yellow),("white",mask_white),("orange",mask_orange),("purple",mask_purple)]

#Merging the two masks together
mask=cv2.bitwise_or(mask_black,mask_violet)
final_mask=cv2.bitwise_not(mask)


#Defining the priorities
priority={("red","circle"):9,("red","star"):3,("red","square"):6,
          ("yellow","circle"):6,("yellow","star"):2,("yellow","square"):4,
          ("white","circle"):3,("white","star"):1,("white","square"):2,
          ("orange","triangle"):"start",("purple","triangle"):"end"}

#Defining elevation levels
H=hsv[:,:,0]
S=hsv[:,:,1]
V=hsv[:,:,2]

elevation=np.full(H.shape,-1,np.int8)
elevation[(H>=38)&(H<=55)&(S>=150)&(V>=60)&(V<100)]=3
elevation[(H>=38)&(H<=58)&(S>=150)&(V>=110)&(V<165)]=2
elevation[(H>=40)&(H<=60)&(S>=150)&(V>=165)&(V<205)]=1

casualties=[]
##start=None
##end=None

#Finding and Drawing individual shapes and centers using contours
for color_name,mask in masks:
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt)<30:
            continue
        m=cv2.moments(cnt)
        if m["m00"]==0:
            continue
        cX=int(m["m10"]/m["m00"])
        cY=int(m["m01"]/m["m00"])

        perimeter=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.04*perimeter,True)
        vertices=len(approx)

        shape="uk"
        if vertices==3:
            shape="triangle"
        elif vertices==4:
            shape="square"
        elif vertices==5 or vertices==10:
            shape="star"
        elif vertices>5:
            shape="circle"
            
        if color_name=="orange" and shape=="triangle":
            start=(cX,cY)
        elif color_name=="purple" and shape=="triangle":
            end=(cX,cY)
        elif (color_name,shape) in priority:
            age_group={"circle":"Children","star":"Adults","square":"Senior Citizens"}
            severity={"red":"Critical","yellow":"Moderate","white":"Safe"}
            casualties.append({"center":(cX,cY),"priority":priority[color_name,shape],"age":age_group[shape],"severity":severity[color_name],"visited":False,"elevation":-1})
                  
        #cv2.drawContours(output,[cnt],-1,(0,255,0),2)
        #cv2.circle(output,(cX,cY),4,(0,0,0),-1)

final_mask[start[1],start[0]]=255
final_mask[end[1],end[0]]=255

#Finding elevation levels
for c in casualties:
    x,y=c["center"]
    radius=12
    x1=max(0,x-radius)
    x2=min(img.shape[1],x+radius+1)
    y1=max(0,y-radius)
    y2=min(img.shape[0],y+radius+1)
    yy,xx=np.ogrid[y1:y2,x1:x2]

    distance=np.sqrt((xx-x)**2+(yy-y)**2)
    ring=((distance>=5)&(distance<=radius))
    local=elevation[y1:y2,x1:x2]

    values=local[ring]
    values=values[values>=0]

    if len(values)>0:
        counts=[np.sum(values==0),np.sum(values==1),np.sum(values==2),np.sum(values==3)]
        c["elevation"]=int(np.argmax(counts))
    else:
        c["elevation"]=0

#Finding the route
current=start
total_distance=0
path_score=0
e0_distance=0
e1_distance=0
e2_distance=0
e3_distance=0
directions=[(-1,-1,math.sqrt(2)),(0,-1,1),(1,-1,math.sqrt(2)),(-1,0,1),(1,0,1),(-1,1,math.sqrt(2)),(0,1,1),(1,1,math.sqrt(2))]

while True:
    unvisited=[c for c in casualties if not c["visited"]]

    #Selecting target
    if len(unvisited)>0:
        candidates=[]
        for c in unvisited:
            cx,cy=c["center"]
            d=math.hypot(cx-current[0],cy-current[1])
            candidates.append((d,c))
        candidates.sort(key=lambda z:z[0])

        if len(candidates)>=2:
            c1=candidates[0][1]
            c2=candidates[1][1]
            d12=math.hypot(c1["center"][0]-c2["center"][0],c1["center"][1]-c2["center"][1])
            if d12<300:
                if c1["priority"]>=c2["priority"]:
                    target_casualty=c1
                else:
                    target_casualty=c2
            else:
                target_casualty=c1
        else:
            target_casualty=candidates[0][1]
        target=target_casualty["center"]
        is_casualty=True
    else:
        target=end
        is_casualty=False
    #
    open_list=[]
    heapq.heappush(open_list,(0,current))
    g_score={current:0}
    parent={}
    found=False

    while open_list:
        _,node=heapq.heappop(open_list)
        if node==target:
            found=True
            break
        x,y=node

        for dx,dy,cost in directions:
            nx=x+dx
            ny=y+dy
            if nx<0 or nx>=img.shape[1]:
                continue
            if ny<0 or ny>=img.shape[0]:
                continue
            if final_mask[ny,nx]==0:
                continue
            if dx!=0 and dy!=0:
                if final_mask[y,nx]==0:
                    continue
                if final_mask[ny,x]==0:
                    continue
            neighbour=(nx,ny)
            new_g=(g_score[node]+cost)
            if (neighbour not in g_score or new_g<g_score[neighbour]):
                g_score[neighbour]=new_g
                parent[neighbour]=node
                h=math.hypot(nx-target[0],ny-target[1])
                heapq.heappush(open_list,(new_g+h,neighbour))
    if not found:
        print("No valid path found")
        exit()

    path=[target]
    node=target
    while node!=current:
        node=parent[node]
        path.append(node)
    path.reverse()

    #process
    for i in range(1,len(path)):
        x1,y1=path[i-1]
        x2,y2=path[i]
        d=math.hypot(x2-x1,y2-y1)

        total_distance+=d
        level=elevation[y2,x2]
        if level==0:
            e0_distance+=d
        elif level==1:
            e1_distance+=d
        elif level==2:
            e2_distance+=d
        elif level==3:
            e3_distance+=d
        cv2.line(output,(x1,y1),(x2,y2),(0,0,0),2)
    current=target

    #Casualties
    if is_casualty:
        displacement=math.hypot(target[0]-start[0],target[1]-start[1])
        casualty_score=(displacement/total_distance)*target_casualty["priority"]
        path_score+=casualty_score
        target_casualty["visited"]=True
        #cv2.circle(output,target,7,(0,0,0),-1)
    else:
        break

#Outputs
coords=[c["center"] for c in casualties]
priority_scores=[c["priority"] for c in casualties]
e_levels=[c["elevation"] for c in casualties]
ages=[c["age"] for c in casualties]
severity=[c["severity"] for c in casualties]
time=((e0_distance/20)+(e1_distance/15)+(e2_distance/10)+(e3_distance/10))

print("Total number of casualties: ",len(casualties))
print("Casualty coordinates: ",coords)
print("Age groups: ",ages)
print("Severities: ",severity)
print("Priority scores: ",priority_scores)
print("Elevation levels: ",e_levels)
print("Path score: ",path_score)
print("Total time: ",time," seconds")

cv2.imshow("image",img)
cv2.imshow("final_mask",final_mask)
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("mask1.png",final_mask)
#cv2.imwrite("output.jpg",output)
