#mean
l=[10,20,30,80,50,70,50]
n=len(l)
s=0
for i in range(0,n):
    s=s+l[i]
mean=s/n
print("Mean is ",mean)
#median
l=[10,20,30,80,50,70,50]
l.sort()
n=len(l)
if n%2==0:
    m1=l[n//2]
    m2=l[(n//2)-1]
    medain=(m1+m2)//2
else:
    medain=l[n//2]
print(medain)

#mode
l=[10,20,30,80,50,70,50]
unique=[]
mod=[]
for i in l:
    if i not in unique:
        unique.append(i)
    else:
        mod.append(i)
print(set(mod))
#variance
l=[10,20,30,80,50,70,50]
l1=[]
n=len(l)
s=0
s1=0
for i in range(0,n):
    s=s+l[i]
mean=s/n
for i in range(0,n):
    l1.append(l[i]-mean)
m=len(l1)
for j in range(0,m):
    l1[j]=l1[j]**2
    s1=s1+l1[j]
variance=s1/n
print(variance)

