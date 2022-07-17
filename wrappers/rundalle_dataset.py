class RuDalleDataset(Dataset):
	clip_filter_thr = 0.24

	def __init__(
			self,
			file_path,
			csv_path,
			tokenizer,
			resize_ratio=0.75,
			shuffle=True,
			load_first=None,
			caption_score_thr=0.6
	):

		self.text_seq_length = model.get_param('text_seq_length')
		self.tokenizer = tokenizer
		self.target_image_size = 256
		self.image_size = 256
		self.samples = []

		self.image_transform = T.Compose([
			T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
			T.RandomResizedCrop(self.image_size,
								scale=(1., 1.),  # в train было scale=(0.75., 1.),
								ratio=(1., 1.)),
			T.ToTensor()
		])

		df = pd.read_csv(csv_path)
		for caption, f_path in zip(df['caption'], df['name']):
			if os.path.isfile(f'{file_path}/{f_path}'):
				# Note: You may want to perform a translation here on the caption... I don't see a difference
				self.samples.append([file_path, f_path, caption])
		if shuffle:
			np.random.shuffle(self.samples)
			print('Shuffled')

	def __len__(self):
		return len(self.samples)

	def load_image(self, file_path, img_name):
		image = PIL.Image.open(f'{file_path}/{img_name}')
		return image

	def __getitem__(self, item):
		item = item % len(self.samples)  # infinite loop, modulo dataset size
		file_path, img_name, text = self.samples[item]
		try:
			image = self.load_image(file_path, img_name)
			image = self.image_transform(image).to(device)
		except Exception as err:  # noqa
			print(err)
			random_item = random.randint(0, len(self.samples) - 1)
			return self.__getitem__(random_item)
		text = tokenizer.encode_text(text, text_seq_length=self.text_seq_length).squeeze(0).to(device)
		return text, image