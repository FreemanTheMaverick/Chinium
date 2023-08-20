from sys import argv

R=float(argv[1])
a1=float(argv[2])
a2=float(argv[3])
a3=float(argv[4])
a4=float(argv[5])

N=50
os=[6,38,86,194,86]
ns=[0,0,0,0,0]

for i in range(1,51):
	r=R*(i/(N+1-i))**2
	if r<R*a1:
		ns[0]+=1
	elif r<R*a2:
		ns[1]+=1
	elif r<R*a3:
		ns[2]+=1
	elif r<R*a4:
		ns[3]+=1
	else:
		ns[4]+=1

print(5,50)
print("em",R)

for i in range(5):
	print(os[i],ns[i])
