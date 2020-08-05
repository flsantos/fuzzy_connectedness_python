import numpy as np


class FuzzyConnector:
    
    def get_neighbors(self, c):
        y = (c % self.m_pixelsPerSlice) / self.m_width
        x = (c % self.m_pixelsPerSlice) % self.m_width
        
        result = []
        result.append(c+1 if x < self.m_width-1 else -1)
        result.append(c-1 if x > 0 else -1)
        result.append(c+self.m_width if y < self.m_height-1 else -1)
        result.append(c-self.m_width if y > 0 else -1)
        
        return np.array(result)
        
    
    def ave(self, c, d):
        #print(self.m_imagePixels[c], self.m_imagePixels[d])
        return 0.5 * (self.m_imagePixels[c] + self.m_imagePixels[d])
    
    def reldiff(self, c, d):
        fc = self.m_imagePixels[c]
        fd = self.m_imagePixels[d]
        
        return 0 if fc == -fd else ((abs(fc - fd)) / (fc + fd))
        
    def walfords_std(self,samples):
        N=len(samples)
        sum = 0
        sumsq = 0
        for x in samples:
            sum = sum + x
            sumsq = sumsq + x**2
        mean = sum/N 
        return np.sqrt((sumsq - N*mean**2)/(N-1))
    
    
    
    #Sets the values of m_mean_ave, m_sigma_ave, m_mean_reldiff, m_sigma_reldiff
    #to the corresponding mean and standard deviation values of ave and reldiff between
    #all combinations of neighbors around all seeds (3x3 centered on each)
    def calculate_means_and_sigmas(self):
        # use a set to store the keys, to avoid get duplicated spels
        spels=set()
        
        for s in self.m_seeds:
            neighbors = self.get_neighbors(s)
            for j in neighbors:
                if (j==-1):
                    continue
                
                neighborsNeighbors = self.get_neighbors(j)
                for k in neighborsNeighbors:
                    if (k==-1):
                        continue
                    spels.add(k)
        
        spels_array = np.array(list(spels))
        
        numSpels = len(spels)
        num_combinations = (numSpels * (numSpels - 1)) / 2
        
        aves = []
        reldiffs = []
        
        for i in range(0,numSpels):
            for j in range(i+1, numSpels):
                #print(i,j, spels_array[i], spels_array[j], self.ave(spels_array[i],spels_array[j]))
                aves.append(self.ave(spels_array[i],spels_array[j]))
                reldiffs.append(self.reldiff(spels_array[i],spels_array[j]))
        
        
        
        self.m_mean_ave=np.mean(aves)
        self.m_mean_reldiff=np.mean(reldiffs)
        self.m_sigma_ave=self.walfords_std(aves)
        self.m_sigma_reldiff=self.walfords_std(reldiffs)
        
        #print("ave mean: " , self.m_mean_ave);
        #print("ave sigma: " , self.m_sigma_ave);
        #print("reldiff mean: " , self.m_mean_reldiff);
        #print("reldiff sigma: " , self.m_sigma_reldiff);
        
    def gaussian(self, val, avg, sigma):
        # fix for when region has no variation (gold standard image)
        if sigma == 0:
            sigma=0.000001
            
        return np.exp(-(1.0/(2.0*sigma*sigma)) * (val - avg) * (val - avg))
        
    def affinity(self, c, d):
        #print(self.ave(c,d))
        #print(self.m_mean_ave)
        #print(self.m_sigma_ave)
        #print(self.ave(c,d), self.m_mean_ave, self.m_sigma_ave)
        g_ave = self.gaussian(self.ave(c,d), self.m_mean_ave, self.m_sigma_ave)
        g_reldiff = self.gaussian(self.reldiff(c,d), self.m_mean_reldiff, self.m_sigma_reldiff)

        
        return min(g_ave, g_reldiff)
        
        

    def __init__(self,img, seeds, threshold):
        ############################# init values ######################
        self.img=None
        self.m_height=None
        self.m_width=None

        self.m_pixelsPerSlice=None

        self.m_mean_ave=None
        self.m_mean_reldiff=None
        self.m_sigma_ave=None
        self.m_sigma_reldiff=None

        self.m_threshold=None

        self.m_imagePixels=None
        self.m_conScene=None

        self.m_seeds=[]
        
        ##Simple Queue
        self.Q = []
        #################################################################
        
        self.img = img.astype(np.int64, casting='unsafe')
        self.m_threshold  = threshold
        
        self.m_height= img.shape[0]
        self.m_width= img.shape[1] 
        
        self.m_pixelsPerSlice = self.m_height * self.m_width
        
        
        self.m_imagePixels = self.img.reshape((-1))
        self.m_conScene = np.zeros(self.m_imagePixels.shape[0])
        
        for s in seeds:
            seed_idx = s[0] + s[1]*self.m_width
            self.m_seeds.append(seed_idx)
            
            ### Initialize seeds with 1.0
            self.m_conScene[seed_idx]=1.0
        self.m_seeds = np.array(self.m_seeds)
        
        
        self.calculate_means_and_sigmas()
        
    ### Algorihtm Label Setting ( like version v. 1000 but with Queue instead of d-aryHeap of "Fuzzy-connected 3D image segmentationat interactive speeds", Udupa et al)
    def run(self):
        
        # Start Algorihtm (Label Setting)
        
        #Puh all seeds o to Q
        
        for s in self.m_seeds:
            self.Q.append(s)
            
        
        while len(self.Q) > 0:
            
            
            # Pop(0) == "FIFO"
            c = self.Q.pop(0)
            
            neighbors = self.get_neighbors(c)
            
            
            for e in neighbors:
                
                # get -1 in case its in the image edges
                if e==-1:
                    continue
                    
                aff_c_e = self.affinity(c, e)
                

                if aff_c_e < self.m_threshold:
                    continue
                
                f_min = min(self.m_conScene[c], aff_c_e)
                
                if f_min > self.m_conScene[e]:
                    self.m_conScene[e] = f_min
                    
                    if e in self.Q:
                        #atualzia o lugar na fila: remove e vai pro final
                        self.Q.remove(e)
                        self.Q.append(e)
                    else:
                        self.Q.append(e)

        return self.m_conScene
    
    
    
    
    
    ### Algorihtm Label Correcting ( v. 0002 of "Fuzzy-connected 3D image segmentationat interactive speeds", Udupa et al)
    def run2(self):
        
        # Start Algorihtm (Label Correcting)
        
        #Puh all seeds c that has affinity greater than 0 with a seed to Q
        for s in self.m_seeds:
            
            neighbors = self.get_neighbors(s)
            for c in neighbors:
                if c==-1:
                    continue
                    
                aff_s_c = self.affinity(s, c)
                if aff_s_c > 0:
                    self.Q.append(c)
            
        
        while len(self.Q) > 0:
            
            
            # Pop(0) == "FIFO"
            c = self.Q.pop(0)
            
            neighbors = self.get_neighbors(c)
            for d in neighbors:
                if d==-1:
                    continue
                
                aff_c_d = self.affinity(c, d)
                
                if aff_c_d < self.m_threshold:
                    continue
                    
                f_min = min(self.m_conScene[d], aff_c_d)
                if f_min > self.m_conScene[c]:
                    self.m_conScene[c] = f_min
                    
                    neighbors = self.get_neighbors(c)
                    for e in neighbors:
                        if e==-1:
                            continue
                        aff_c_e = self.affinity(c, e)
                        #conditions of section 3.3.1
                        #if aff_c_e >0:  
                        #if f_min > self.m_conScene[e]:
                        if f_min > self.m_conScene[e] and aff_c_e >= self.m_conScene[e]:
                            self.Q.append(e)
            

        return self.m_conScene
        
       
    