import os
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F


#pytorch 1.2.0 implementation


class generator(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)
		self.forward = self.forward_

	def forward(self, pointz):
		return self.forward_(pointz.unsqueeze(0)).squeeze(0)

	def forward_(self, pointz, x=None):
		'''pointz is [1,N,D+L]'''

		if x is None:
			x = self.linear_1(pointz)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_2(x)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_3(x)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_4(x)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_5(x)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_6(x)
		x = F.leaky_relu(x, negative_slope=0.02, inplace=True)

		x = self.linear_7(x)

		#x = torch.clamp(x, min=0, max=1)
		x = torch.max(torch.min(x, x*0.01+0.99), x*0.01)
		
		return x
	
	def forward_intermediate(self, pointz, return_after_layer=7, return_after_activation=True):
		x = pointz

		layers = [
			self.linear_1,
			self.linear_2,
			self.linear_3,
			self.linear_4,
			self.linear_5,
			self.linear_6,
		]
		for l, layer in enumerate(layers):
			x = layer(x)
			if l+1==return_after_layer and not return_after_activation: return x
			x = F.leaky_relu(x, negative_slope=0.02, inplace=False) ## NOTE: inplace=False is needed by jvp in feature tracking
			if l+1==return_after_layer and return_after_activation: return x

		x = self.linear_7(x)
		if return_after_activation: return x
		x = torch.max(torch.min(x, x*0.01+0.99), x*0.01)
		return x
		
	
	def forward_chunk(self, pointz):
		out = torch.empty([len(pointz),1], device=pointz.device)
		chunks = torch.split(pointz, int(1e5))
		print(len(chunks))
		i = 0
		for chunk in chunks:
			print(i)
			out[i:i+len(chunk)] = self.forward(chunk)
			i += len(chunk)
		return out

	def forward_z(self, points, z):
		out = torch.zeros([len(points),1], device=points.device)
		chunks = torch.split(points, int(1e5))
		l1_z = z@self.linear_1.weight[:,3:].T
		i = 0
		for chunk in chunks:
			x = chunk@self.linear_1.weight[:,:3].T + l1_z
			x += self.linear_1.bias ## dont forget the bias :)
			out[i:i+len(chunk)] = self.forward_(None, x.unsqueeze(0)).squeeze(0)
			i += len(chunk)
		return out


class encoder(nn.Module):
	def __init__(self, ef_dim, z_dim, device=None):
		super(encoder, self).__init__()
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.ef_dim = ef_dim
		self.z_dim = z_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.in_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)

	def forward(self, inputs, is_training=False):
		d_1 = self.in_1(self.conv_1(inputs))
		d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

		d_2 = self.in_2(self.conv_2(d_1))
		d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)
		
		d_3 = self.in_3(self.conv_3(d_2))
		d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

		d_4 = self.in_4(self.conv_4(d_3))
		d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.z_dim)
		d_5 = torch.sigmoid(d_5)

		return d_5


class im_network(nn.Module):
	def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
		super(im_network, self).__init__()
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.encoder = encoder(self.ef_dim, self.z_dim)
		self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)

	def forward(self, inputs, z_vector, point_coord, is_training=False):
		if is_training:
			z_vector = self.encoder(inputs, is_training=is_training)
			zs = z_vector.view(-1,1,self.z_dim).repeat(1,point_coord.size()[1],1)
			pointz = torch.cat([point_coord,zs],2)
			net_out = self.generator.forward_(pointz)
		else:
			if inputs is not None:
				z_vector = self.encoder(inputs, is_training=is_training)
			if z_vector is not None and point_coord is not None:
				zs = z_vector.view(-1,1,self.z_dim).repeat(1,point_coord.size()[1],1)
				pointz = torch.cat([point_coord,zs],2)
				net_out = self.generator.forward_(pointz)
			else:
				net_out = None

		return z_vector, net_out


class IM_AE(object):
	def __init__(self, config):
		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = config.sample_vox_size
		if self.sample_vox_size==16:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==32:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==64:
			self.load_point_batch_size = 16*16*16*4
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		self.input_size = 64 #input voxel grid size

		self.ef_dim = 32
		self.gf_dim = 128
		self.z_dim = 256
		self.point_dim = 3

		self.dataset_name = config.dataset
		self.checkpoint_dir = config.checkpoint_dir


		if 'device' not in config:
			if torch.cuda.is_available():
				self.device = torch.device('cuda')
				torch.backends.cudnn.benchmark = True
			else:
				self.device = torch.device('cpu')
		else:
			self.device = config.device

		#build model
		self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
		self.im_network.to(self.device)
		#print params
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 2
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='IM_AE.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0


		#keep everything a power of 2
		self.cell_grid_size = 4
		self.frame_grid_size = 64
		self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
		self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
		self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change


		#get coords for testing
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
		self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
		self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
		for i in range(dimc):
			for j in range(dimc):
				for k in range(dimc):
					self.cell_x[i,j,k] = i
					self.cell_y[i,j,k] = j
					self.cell_z[i,j,k] = k
		for i in range(dimf):
			for j in range(dimf):
				for k in range(dimf):
					self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
					self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
					self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
					self.frame_coords[i,j,k,0] = i
					self.frame_coords[i,j,k,1] = j
					self.frame_coords[i,j,k,2] = k
					self.frame_x[i,j,k] = i
					self.frame_y[i,j,k] = j
					self.frame_z[i,j,k] = k
		self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.real_size-0.5
		self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
		self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
		self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
		self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
		self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
		self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
		self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
		self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/dimf-0.5
		self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])
		
		self.sampling_threshold = 0.5 #final marching cubes threshold

	@property
	def model_dir(self):
		return "{}_ae_{}".format(self.dataset_name, self.input_size)

	def z2voxel(self, z):
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		
		frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
		queue = []
		
		frame_batch_num = int(dimf**3/self.test_point_batch_size)
		assert frame_batch_num>0
		
		#get frame grid values
		for i in range(frame_batch_num):
			point_coord = self.frame_coords[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			point_coord = np.expand_dims(point_coord, axis=0)
			point_coord = torch.from_numpy(point_coord)
			point_coord = point_coord.to(self.device)
			_, model_out_ = self.im_network(None, z, point_coord, is_training=False)
			model_out = model_out_.detach().cpu().numpy()[0]
			x_coords = self.frame_x[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			y_coords = self.frame_y[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			z_coords = self.frame_z[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])
		
		#get queue and fill up ones
		for i in range(1,dimf+1):
			for j in range(1,dimf+1):
				for k in range(1,dimf+1):
					maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					if maxv!=minv:
						queue.append((i,j,k))
					elif maxv==1:
						x_coords = self.cell_x+(i-1)*dimc
						y_coords = self.cell_y+(j-1)*dimc
						z_coords = self.cell_z+(k-1)*dimc
						model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
		
		# print("running queue:",len(queue))
		cell_batch_size = dimc**3
		cell_batch_num = int(self.test_point_batch_size/cell_batch_size)
		assert cell_batch_num>0
		#run queue
		while len(queue)>0:
			batch_num = min(len(queue),cell_batch_num)
			point_list = []
			cell_coords = []
			for i in range(batch_num):
				point = queue.pop(0)
				point_list.append(point)
				cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
			cell_coords = np.concatenate(cell_coords, axis=0)
			cell_coords = np.expand_dims(cell_coords, axis=0)
			cell_coords = torch.from_numpy(cell_coords)
			cell_coords = cell_coords.to(self.device)
			_, model_out_batch_ = self.im_network(None, z, cell_coords, is_training=False)
			model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
			for i in range(batch_num):
				point = point_list[i]
				model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
				x_coords = self.cell_x+(point[0]-1)*dimc
				y_coords = self.cell_y+(point[1]-1)*dimc
				z_coords = self.cell_z+(point[2]-1)*dimc
				model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
				
				if np.max(model_out)>self.sampling_threshold:
					for i in range(-1,2):
						pi = point[0]+i
						if pi<=0 or pi>dimf: continue
						for j in range(-1,2):
							pj = point[1]+j
							if pj<=0 or pj>dimf: continue
							for k in range(-1,2):
								pk = point[2]+k
								if pk<=0 or pk>dimf: continue
								if (frame_flag[pi,pj,pk] == 0):
									frame_flag[pi,pj,pk] = 1
									queue.append((pi,pj,pk))
		return model_float

if __name__=="__main__":
	import h5py
	
	class DotDict(dict):
		__getattr__ = dict.__getitem__
		__setattr__ = dict.__setitem__
		__delattr__ = dict.__delitem__

	## Most of these are not actually used but we need them to initialize the model
	FLAGS = DotDict()
	FLAGS.sample_dir = "samples/im_ae_out"
	FLAGS.sample_vox_size = 64
	FLAGS.dataset = "all_vox256_img"
	FLAGS.checkpoint_dir = "checkpoint"
	FLAGS.device = "cpu" ## optional

	## Load model
	im_ae = IM_AE(FLAGS)
	model_dir = "IM_NET/checkpoint/all_vox256_img_ae_64/IM_AE.model64-399.pth"
	im_ae.im_network.load_state_dict(torch.load(model_dir, map_location=im_ae.device))
	im_ae.im_network.eval()
	model = im_ae.im_network.generator
	device = im_ae.device

	## Load latent codes
	filename = "IM_NET/checkpoint/all_vox256_img_ae_64/all_vox256_img_train_z.hdf5"
	with h5py.File(filename, "r") as f:
		batch_z = f['zs'][()]  # np.array
	batch_z = torch.from_numpy(batch_z)
	batch_z = batch_z.to(device)

	z = batch_z[11945]
	V = im_ae.z2voxel(z)