import numpy as np
import h5py
import matplotlib.pyplot as plt

h5=h5py.File('run_juno_emcee_cpcs_parallel_5000.h5', 'r')
chain = np.array(h5['mcmc']['chain'][:])
h5.close()

h5=h5py.File('run_regenerate_Tb_profile_parallel_cpcs_TB_5000.h5', 'r')
tbs = np.array(h5['tb'][:,0,:])
# tbs = np.array(h5['tb'])
h5.close()

nstep=5000

shape = tbs.shape
print(shape)
# print(shape[1])
# print(shape[2])

# for iwalker in range(shape[1]):
#     for istep in range(1,shape[0]):
#         nh3,temp,rhmax =chain[istep,iwalker,:]
#         nh3_1,temp_1,rhmax_1 =chain[istep-1,iwalker,:]       
 
#         ## if the new step not accepted, discard the newest tb simulation
#         if nh3==nh3_1 and temp==temp_1 and rhmax==rhmax_1 :
#             tbs[istep,iwalker,:]=tbs[istep-1,iwalker,:]

# obs=np.array([474.58510281, 469.42513666, 454.00808555, 428.59559118, 
#               338.13016122, 335.65949356, 328.01197674, 314.60534003,
#               251.9730167 , 250.46642377, 245.71888005, 237.15115289, 
#               194.47971955, 193.67185714, 191.10407859, 186.40702401,
#               141.18445694, 141.06723252, 140.59821156, 139.46693178,
#             ])

obs=np.array([478.85214549, 333.24137522, 253.26389628, 196.52085143, 142.22266099])

anomaly=np.array([0. ,      0.7181285 ,2.1207075 ,2.969638,  1.1837736])

tbbgd=obs-anomaly

residual_0=tbs-tbbgd

# chisq=np.array([0]*300)
# for istep in range(300):
#     chisq[istep]=np.sum((residual_0[istep,:]/matrix[istep,:]/0.02)**2)
print(residual_0.shape)
# print(matrix.shape)

fig, ax = plt.subplots(5, 1,figsize=(8, 8), dpi=300)
for i in range(5):
    print(i)
    ax[i].plot(residual_0[:,i],label=r"$0^o$", color="gray", linewidth=1, alpha=0.5)
    # ax[i].plot(residual_0[:,i*4+1],label=r"$15^o$")
    # ax[i].plot(residual_0[:,i*4+2],label=r"$30^o$")
    # ax[i].plot(residual_0[:,i*4+3],label=r"$45^o$")
    ax[i].set_xlim([0,nstep])
    ax[i].set_ylabel(f"CH{i+2}")
    # ax[i].set_xticklabels([])
    ax[i].grid()
    mean=np.mean(residual_0[200:nstep,i]-anomaly[i]) 
    max1=max(residual_0[:,i]-anomaly[i]) 
    ax[i].text(2000,anomaly[i]+0.5, f"bias={mean: 4.2f} K", fontsize=10, ha='center', va='center', color='blue')
    ax[i].axhline(y=anomaly[i], color='r', linestyle='--')
    ax[i].text(200,anomaly[i]+0.5, f"{anomaly[i]: 4.2f} K", fontsize=10, ha='center', va='center', color='red')
    # ax[0].set_ylim([0,50])
    # ax[1].set_ylim([0,20])
    # ax[2].set_ylim([0,15])
    # ax[3].set_ylim([0,10])
    ax[i].set_ylim([-2,5])
    ax[4].set_xlabel("MCMC Step")
    # ax[4].set_xticks([0,50,100,150,200,250,300])
    # ax[4].set_xticklabels([0,50,100,150,200,250,300])
    ax[0].set_title("CPCs anomaly [K]",loc="left")
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=4)
plt.savefig("TB_cpcs_noch2_1000.png")




# plt.figure(figsize=(6,2),dpi=300)
# plt.plot(chisq[:4980])
# plt.xlim([0,5000])
# plt.ylabel(f"chi_Sq")
# # plt.xticks([])
# plt.grid()
# plt.ylim([0,100])
    # ax[0].set_ylim([-10,40])
    # ax[1].set_ylim([-20,20])
    # ax[2].set_ylim([-15,15])
    # ax[3].set_ylim([-10,10])
    # ax[4].set_ylim([-4,0])

# plt.savefig("chisq_5000.png")



# h5=h5py.File('TB_cpcs_1000.h5', 'r')
# tbs = np.array(h5['tb'][:,:,:])
# h5.close()

# c1=np.array([0.]*5)

# c1[0]=obs[0]-obs[3]
# c1[1]=obs[4]-obs[7]
# c1[2]=obs[8]-obs[11]
# c1[3]=obs[12]-obs[15]
# c1[4]=obs[16]-obs[19]

# fig, ax = plt.subplots(5, 1,figsize=(8, 8),dpi=300)
# for i in range(5):
#     ax[i].plot((tbs[:,0,i*4]-tbs[:,0,i*4+3]),'b-')
#     # ax[i].plot((tbs[:,1,i*4]-tbs[:,1,i*4+3]),'g-')
#     # ax[i].plot((tbs[:,2,i*4]-tbs[:,2,i*4+3]),'y-')
#     # ax[i].plot((tbs[:,3,i*4]-tbs[:,3,i*4+3]),'m-')
#     # ax[i].plot((tbs[:,4,i*4]-tbs[:,4,i*4+3]),'k-')

#     mean=np.mean(tbs[200:nstep,0,i*4]-tbs[200:nstep,0,i*4+3]) 
#     max1=max(tbs[:,0,i*4]-tbs[:,0,i*4+3]) 
#     ax[i].text(700,max1, f"Mean bias {mean- c1[i]: 6.2f} K ", fontsize=12, ha='center', va='center', color='red')

#     print(c1[i])
#     ax[i].axhline(y=c1[i], color='r', linestyle='--')
#     ax[i].set_xlim([0,nstep])
#     ax[i].set_ylabel(f"CH{i+2}")
#     # ax[i].set_xticks([0,50,100,150,200,250,300])
#     # ax[i].set_xticklabels([])
#     ax[i].grid()

# ax[0].set_title(r"Limb darkening (45$^o$)",loc="left")



# # .ylim([0,20])
# ax[0].set_ylim([0,90])
# ax[1].set_ylim([0,40])
# ax[2].set_ylim([0,30])
# ax[3].set_ylim([6,15])
# ax[4].set_ylim([1.5,2.2])
# ax[4].set_xlabel("MCMC Step")
# # ax[4].set_xticklabels([0,50,100,150,200,250,300])
# plt.savefig("C1_cpcs_1000.png")