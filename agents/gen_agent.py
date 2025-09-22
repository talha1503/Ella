import sys
from datetime import time

from tools.model_manager import global_model_manager
from .sg.builder.object import AGENT_TAGS
from .memory import SemanticMemory, EpisodicMemory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vico.agents import Agent
from vico.tools.utils import *

class GenAgent(Agent):
	def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
				 lm_source='azure', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, enable_gt_segmentation=True):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		self.lm_source = lm_source
		self.wake_up_hour = None
		self.generator = global_model_manager.get_generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
		# self.generator_4o = global_model_manager.get_generator(lm_source, 'gpt-4o', max_tokens, temperature, top_p, logger)
		self.generator_embedding = global_model_manager.get_generator(lm_source, 'text-embedding-3-small', max_tokens, temperature, top_p, logger)
		self.react_freq = self.scratch["react_freq"] if "react_freq" in self.scratch else 60
		self.generate_new_schedule_freq = 600
		self.last_new_schedule_time = None
		# if self.debug:
		# 	self.react_freq = 300 # 5 min for debug
		self.importance_trigger_curr = self.scratch["importance_trigger_curr"]
		self.importance_trigger_max = self.scratch["importance_trigger_max"]
		self.importance_ele_n = self.scratch["importance_ele_n"]
		self.daily_plan = self.scratch["daily_plan"]
		self.hourly_schedule = self.scratch["hourly_schedule"]
		self.curr_event = self.scratch["act_event"]
		self.last_actions = [None] * 5 # maintain 5 last actions
		self.enable_gt_segmentation = enable_gt_segmentation
		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=-1 if self.enable_gt_segmentation else 2, debug=self.debug, logger=self.logger)
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), lm_source, debug=self.debug, logger=self.logger)
		lastevent = self.e_mem.get_memory(-1)
		if lastevent is not None:
			self.last_react_time = lastevent.event_time
		else:
			self.last_react_time = None
		self.commuting = None
		if self.scratch["act_address"] is None:
			# print("act is None")
			self.curr_goal_address = None
			self.curr_goal_pos = None
			self.curr_goal_bbox = None
			self.curr_goal_description = None
			self.curr_goal_duration = None
			self.curr_goal_end_time = None
		else:
			# print("act is not None")
			curr_goal_dict = self.s_mem.get_knowledge(self.scratch["act_address"])
			self.curr_goal_address = self.scratch["act_address"]
			self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
			self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
			self.curr_goal_description = self.scratch["act_description"]
			self.curr_goal_duration = self.scratch["act_duration"]
			# if "act_start_time" in self.scratch and self.scratch["act_start_time"] is not None:
			if "act_start_time" in self.scratch and self.scratch["act_start_time"] is not None:
				# print("act_start_time is not None")
				self.curr_goal_end_time = datetime.strptime(self.scratch["act_start_time"], "%B %d, %Y, %H:%M:%S") + timedelta(minutes=self.curr_goal_duration)
			else:
				next_schedule_index = self.get_next_schedule_index()
				# print("next_schedule_index", next_schedule_index)
				# print("curr_goal_description", self.curr_goal_description)
				curr_goal_start_time = datetime.combine(self.curr_time.date(), datetime.min.time()) + timedelta(minutes=sum(activity[1] for activity in self.hourly_schedule[:next_schedule_index]))
				# print(f"curr_goal_start_time: {curr_goal_start_time}")
				self.curr_goal_end_time = curr_goal_start_time + timedelta(minutes=self.curr_goal_duration)
				# print(f"curr_goal_end_time: {self.curr_goal_end_time}")
		self.need_react_in_building = True

	def update_action(self, action):
		self.last_actions.pop(0)
		self.last_actions.append(action)

	def reset(self, name, pose):
		super().reset(name, pose)
		if self.scratch["act_address"] is None:
			self.curr_goal_pos = None
			self.curr_goal_bbox = None
		else:
			curr_goal_dict = self.s_mem.get_knowledge(self.scratch["act_address"])
			self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
			self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
		self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=-1 if self.enable_gt_segmentation else 2, debug=self.debug, logger=self.logger)
		self.e_mem = EpisodicMemory(os.path.join(self.storage_path, "episodic_memory"), self.lm_source, debug=self.debug, logger=self.logger)

	def _process_obs(self, obs): # override
		# print(f"{self.name} at {self.curr_time} processing obs...")
		self.cur_objects = self.s_mem.update_ga(obs)
		# print("cur_objects:", self.cur_objects)
		self.current_place = obs['current_place']
		self.obs = obs
		self.curr_time = obs['curr_time']
		if self.obs['new_day'] or self.hourly_schedule == []:
			if len(self.e_mem) == 0: # it is the first day
				self.daily_plan = self.generate_daily_plan()
			else: # it is the new day after first day\
				self.daily_plan = self.generate_daily_plan()
				# self.daily_plan = self.revise_identity_and_generate_daily_plan() # disable it for now
			self.scratch["daily_plan"] = self.daily_plan
			self.logger.debug(f"{self.name}'s daily plan is {self.daily_plan}.")
			os.makedirs(f"output/permanent/{self.storage_path.split('/')[1]}/{self.name}/", exist_ok = True)
			stored_schedule_path = f"output/permanent/{self.storage_path.split('/')[1]}/{self.name}/stored_schedule_path.json"
			if self.debug and os.path.exists(stored_schedule_path):
				self.hourly_schedule = json.load(open(stored_schedule_path, "r"))
			else:
				self.hourly_schedule = self.generate_hourly_schedule()
				with open(stored_schedule_path, 'w') as f:
					json.dump(self.hourly_schedule, f)

			self.logger.info(f"{self.name}'s hourly schedule is {self.hourly_schedule}.")
		self.perceived = []

		if len(self.obs['events']) > 0:
			desc = ""
			for event in self.obs['events']:
				img_path = None
				kws = []
				if event["type"] == "speech":
					kws = []
					# print("event:", event)
					if event["position"][:2] == self.pose[:2]:
						continue
					# print("debug::::eventcontent", event["content"])
					# print("debug::::eventsubject", event["subject"])
					if event["subject"] is None:
						event["subject"] = self.s_mem.get_name_from_position(event["position"])
						print("s_mem.get_name_from_position:", event["subject"])
						kws.append(event["subject"])
					if event["subject"] is not None:
						event["content"] = {"from_name": event["subject"],"utterance": event["content"]}
						# print(f"{self.curr_time}: {event['content']['from_name']} said '{event['content']['utterance']}'")
						# print("event:", event)
						desc = event["content"]["from_name"] + " said '" + event["content"]["utterance"] + "'"
						kws.append(event["subject"])
					else:
						desc = ""
				elif event["type"] == "store goods":
					# print("store goods event:", event)
					desc = f"I saw a shelf with a variety of goods ({event['content']}) in a store"
					kws = re.findall(r"'name': '([^']*)'", event["content"])
				else:
					desc = ""
				if desc != "" and len(kws) > 0:
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.perceived.append(self.e_mem.add_memory(event_type=event["type"], event_time=self.curr_time, event_position=event["position"], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc)))
			if desc != "":
				self.last_react_time = self.curr_time
		
		if self.obs['rgb'] is not None: # if None, means sleeping, no need to process
			# print(f"{self.name} curr chat:", self.e_mem.curr_chat)
			bool1 = self.last_react_time is None
			bool2 = (self.last_react_time != self.curr_time and (self.curr_time - self.last_react_time).total_seconds() % self.react_freq == 0) if bool1 == False else None
			bool3 = ((self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "turn_around_to_sound_source") or (self.last_actions[-2] is not None and self.last_actions[-2]["type"] == "turn_around_to_sound_source")) if bool1 == False and bool2 == False else None
			bool4 = (len(self.e_mem.curr_chat) > 0) if bool1 == False and bool2 == False and bool3 == False else None
			# print(f"{self.name} at {self.curr_time.strftime('%B %d, %Y, %H:%M:%S')}:", bool1, bool2, bool3, bool4)
			if bool1 or bool2 or bool3 or bool4:
				if self.enable_gt_segmentation:
					# s, p, o = self.generate_event_triple_from_rgb(self, rgb=self.obs['rgb']) # spo = subject, predicate, object
					obs_seg_unique_ids = np.unique(self.obs['segmentation']).tolist()
					# print("obs_seg_unique_ids:", obs_seg_unique_ids)
					# img_path = os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
					# Image.fromarray(self.obs['rgb']).save(img_path)
					img_path = None
					# perceived_segnames = []
					for seg_id in obs_seg_unique_ids:
						if seg_id != -1:
							seg_id_count = np.count_nonzero(self.obs['segmentation'] == seg_id)
							# print(f"{self.name} seg name:", self.obs["gt_seg_entity_idx_to_info"][seg_id])
							#  print(f"{self.name} seg count:", seg_id_count)
							if (seg_id_count / np.array(self.obs['segmentation']).size) > 0.001: # if the segmentation mask has at least 5% of pixels, then we put it into the memory
								# if self.obs["gt_seg_entity_idx_to_info"] is not None and seg_id is not None:
								seg_info = self.obs["gt_seg_entity_idx_to_info"][seg_id]
								# print("seg_info:", seg_info)
								seg_type = seg_info["type"]
								# print(f"{self.name} obs_seg_unique_ids:", obs_seg_unique_ids)
								# print(f"{self.name} seg type:", seg_type)
								seg_name = seg_info["name"]
								kws = [seg_type + ':' + seg_name, "", ""]
								desc = f"I saw {seg_name}"
								# desc = self.generator_4o.generate(f"Describe what you see in one sentence in the past tense. For example, A man was driving a car.", img=img_path)
								# print("goes here 3:", desc)
								# desc = ' '.join(kws)
								# img_path = seg_type
								if desc not in self.e_mem.experience_embedding.keys():
									embedding = self.generator_embedding.get_embedding(desc, caller="embed_observation_description")
								else:
									embedding = self.e_mem.experience_embedding[desc]
								# print("goes here 4:", embedding)
								# print("debug:np.array(self.obs['segmentation']).size:", np.array(self.obs['segmentation']).size)
								# print("debug:seg_proportion:", seg_id_count/np.array(self.obs['segmentation']).size)
								self.perceived.append(self.e_mem.add_memory(event_type="observation", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc)))
					# print(f"{self.name} at {self.curr_time} perceived_segnames:", perceived_segnames)
				else:
					img_path = None
					# img_path = os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
					# Image.fromarray(self.obs['rgb']).save(img_path)
					for cur_obj in self.cur_objects:
						seg_type = "avatar" if cur_obj["tag"] in AGENT_TAGS else cur_obj["tag"]
						# print(f"{self.curr_time} seg_type:", seg_type, "seg_name:", cur_obj["name"])
						seg_name = cur_obj["name"] # This is where to get the name of the avatar/objects in the event keywords
						kws = [seg_type + ':' + seg_name, "", ""]
						desc = f"I saw {seg_name}"
						if desc not in self.e_mem.experience_embedding.keys():
							embedding = self.generator_embedding.get_embedding(desc, caller="embed_observation_description")
						else:
							embedding = self.e_mem.experience_embedding[desc]
						self.perceived.append(self.e_mem.add_memory(event_type="observation", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=kws, event_img=img_path, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc)))
				self.last_react_time = self.curr_time
				self.e_mem.save_memory()
		for event in self.perceived:
			self.importance_trigger_curr -= event.event_poignancy
			self.importance_ele_n += 1
		# must put below after perception, otherwise the agent will not percept at all.
		if self.obs['action_status'] == "FAIL":
			self.logger.info(f"{self.name} failed to execute last action.")
			if self.last_actions[-1] is not None:
				if self.last_actions[-1]["type"] == 'converse':
					print(f"{self.name} action status is fail so dropping the following chat: {self.e_mem.curr_chat[-1]}")
					self.e_mem.curr_chat.pop()
					self.update_action(None)
	
	def get_focused_event(self, retrieved):
		# No "self events"
		copy_retrieved = retrieved.copy()

		for index, retrieved_single in enumerate(copy_retrieved): 
			curr_event = retrieved_single["curr_event"]
			# print("event type:", curr_event.event_type)
			# print("event description:", curr_event.event_description)
			if curr_event.event_keywords[0] == self.name:
				retrieved.pop(index)
		
		# Always choose avatar first.
		priority = []
		for retrieved_single in retrieved: 
			curr_event = retrieved_single["curr_event"]
			if (curr_event.event_type == "observation" and curr_event.event_keywords[0].split(':')[0] == "avatar"):
				priority += [retrieved_single]
		if priority:
			return random.choice(priority)

		# Skip idle.
		for retrieved_single in retrieved: 
			curr_event = retrieved_single["curr_event"]
			if "is idle" not in curr_event.event_description: 
				priority += [retrieved_single]
		if priority:
			return random.choice(priority)
		
		return None
	
	def reflection_trigger(self):
		# print("current importance trigger:", self.importance_trigger_curr)
		if self.importance_trigger_curr <= 0 and [] != self.e_mem.experience:
			return True
		return False
	
	def generate_focal_points(self, num=3):
		events = [[event.event_last_access_time, event] for event in self.e_mem.experience if "idle" not in event.event_description]
		events = sorted(events, key=lambda x: x[0])
		events = [event for _, event in events]

		statements = ""
		for event in events[-1*self.importance_ele_n:]: 
			statements += event.event_description + "\n"
		
		with open ("generative_agents/persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", str(num))
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'
		special_instruction = "Output must be a list of str."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_focal_points")
		try:
			response = "1) " + response.strip()
			ret = []
			for i in response.split("\n"): 
				ret += [i.split(") ")[-1]]
			return ret
		except:
			return ["Who am I"] * num
		
	def generate_insights_and_evidence(self, events, num=5):
		statements = ""
		for count, event in enumerate(events): 
			statements += f'{str(count)}. {event["event_description"]}\n'
		
		with open ("generative_agents/persona/prompt_template/v2/insight_and_evidence_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", str(num))
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		# print("prompt:", prompt)
		response = self.generator.generate(prompt, caller="generate_insights_and_evidence")
		# print("response1:", response)
		response = "1. " + response.strip()
		# print("response of generate_insights_and_evidence:", response)
		ret = dict()
		for i in response.split("\n"): 
			row = i.split(". ")[-1]
			# print("row:", row)
			try:
				thought = row.split("(because of ")[0].strip()
				# print("thought:", thought)
				evi_raw = row.split("(because of ")[1].split(")")[0].strip()
				# print("evi_raw1:", evi_raw)
				evi_raw = re.findall(r'\d+', evi_raw)
				# print("evi_raw2:", evi_raw)
				evi_raw = [int(i.strip()) for i in evi_raw]
				# print("evi_raw3:", evi_raw)
				ret[thought] = evi_raw
			except:
				# print("continue")
				continue
		try:
			for thought, evi_raw in ret.items(): 
				evidence_node_id = [events[i].event_id for i in evi_raw]
				ret[thought] = evidence_node_id
				if len(thought.strip()) == 0:
					ret.pop(thought)
			return ret
		except:
			return {"this is blank": "node_1"}
		
	def generate_poig_score(self, description):
		if "is idle" in description:
			return 1
		
		with open ("generative_agents/persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.name)
		prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		prompt = prompt.replace("!<INPUT 3>!", description)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = "5"
		special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_poig_score")
		try:
			response = response.strip()
			return int(response)
		except:
			return 4
	
	def reflect(self):
		focal_points = self.generate_focal_points(num=3)
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 30)
		for focal_point, events in retrieved.items():
			thoughts = self.generate_insights_and_evidence(events, 5)
			for thought, evidence in thoughts.items():
				expiration = self.curr_time + timedelta(days=30)
				s, p, o = self.generate_action_event_triple(thought)
				keywords = [s, p, o]
				thought_poignancy = self.generate_poig_score(thought)
				if thought not in self.e_mem.experience_embedding.keys():
					if thought != "":
						embedding = self.generator_embedding.get_embedding(thought, caller="embed_thought")
					else:
						continue
				else:
					embedding = self.e_mem.experience_embedding[thought]
				img_path = None
				self.e_mem.add_memory(event_type="thought", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.obs['current_place'], event_keywords=keywords, event_img=img_path, event_description=thought, event_text_ft=embedding, event_poignancy=thought_poignancy, event_expiration=expiration)

	def reset_reflection_counter(self):
		self.importance_trigger_curr = self.importance_trigger_max
		self.importance_ele_n = 0

	def get_3d_bounds_center(self, entity_idx_list, segmentation, depth, intrinsic_matrix):
		mask = np.isin(segmentation, entity_idx_list)
		assert np.any(mask)
		ys, xs = np.where(mask)
		depth_values = depth[ys, xs]

		### handle outliers
		mean_depth_values = np.mean(depth_values)
		std_depth_values = np.std(depth_values)
		outlier_mask = (depth_values < mean_depth_values - 3*std_depth_values) | (depth_values > mean_depth_values + 3*std_depth_values)
		normal_mean = np.mean(depth_values[~outlier_mask])
		depth_values[outlier_mask] = normal_mean

		# print(depth_values.tolist())

		fx = intrinsic_matrix[0, 0]
		fy = intrinsic_matrix[1, 1]
		cx = intrinsic_matrix[0, 2]
		cy = intrinsic_matrix[1, 2]
		X = (xs - cx) * depth_values / fx
		Y = (ys - cy) * depth_values / fy
		Z = depth_values
		xmin, xmax = np.min(X), np.max(X)
		ymin, ymax = np.min(Y), np.max(Y)
		zmin, zmax = np.min(Z), np.max(Z)
		center_x = (xmin + xmax) / 2.0
		center_y = (ymin + ymax) / 2.0
		center_z = (zmin + zmax) / 2.0
		center_3d = np.array([center_x, center_y, center_z], dtype=np.float32)
		return center_3d
	
	def last_actions_handler(self):
		# print(f"[{self.curr_time}] {self.name} last_actions:", [action["type"] if action is not None else None for action in self.last_actions])
		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == 'converse':
			return None
		
		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "turn_around_to_sound_source": # here, we assume the sound source must be an avatar
			has_avatar_in_perception = False
			for event_instance in self.perceived:
				if event_instance.event_type == "observation" and "avatar" in event_instance.event_keywords[0]:
					has_avatar_in_perception = True
			if not has_avatar_in_perception:
				print("Turned around but not finding any avatar, hence walking 5m forward")
				action = {
					'type': 'move_forward',
					'arg1': 5,
				}
				return action
		return "continue"
	
	def obs_events_handler(self):
		for event in self.obs["events"]:
			if event["type"] == "speech" and event["content"]["from_name"] != self.name:
				# print(f"{self.name} heard speech: {event['content']['utterance']}")
				sound_source_in_perception = False
				for event_instance in self.perceived:
					# if event_instance.event_type == "observation":
						# print(f"For {self.name}")
						# print("event_instance.event_keywords:", event_instance.event_keywords)
						# print("event['content']['from_name']:", event['content']['from_name'])
						# print('-' * 10)
					if event_instance.event_type == "observation" and event['content']['from_name'] in event_instance.event_keywords[0]:
						sound_source_in_perception = True
				# print('*' * 10)
				# print(f"{self.name} sound source in perception:", sound_source_in_perception)
				self.e_mem.curr_chat += [[event["content"]["from_name"], event["content"]["utterance"]]]
				if not sound_source_in_perception:
					# print("Sound source not in perception.")
					# if self.generate_to_turn(event['content']['utterance']):
						# print("Execute turning around to the sound source...")
					action = self.turn_to_pos(event["position"])
					action_temp = action.copy()
					action_temp['type'] = "turn_around_to_sound_source"
					return action
					# else:
						# print(f"Critical: no turn for {self.name}")
		return "continue"
	
	def focused_event_handler(self, focused_event):
		# print(f"{self.name} focused_event:", focused_event["curr_event"].event_keywords[0].split(':')[0])
		if focused_event["curr_event"].event_type == "observation" and focused_event["curr_event"].event_keywords[0].split(':')[0] == "avatar" and focused_event["curr_event"].event_keywords[0].split(':')[1] != self.name: # focused_event.event_keywords[0] is the subject of the observation (type:name)
			target_knowledge = self.s_mem.get_knowledge(focused_event["curr_event"].event_keywords[0].split(':')[1])
			if self.react_converse(self.obs, focused_event, target_knowledge):
				self.need_react_in_building = True
				experience = self.e_mem.retrieve_latest_memory()
				observation_kws = None
				for event in experience:
					if event["event_type"] == "observation":
						observation_kws = event["event_keywords"]
				if observation_kws == None:
					raise ValueError("observation_kws should not be None, there's a bug somewhere")
				utt, _ = self.converse(self.obs, focused_event, target_agent_state="null") # currently set to "null" because it's hard to use segmentation to infer whether the agent is on the vehicle or not
				if self.enable_gt_segmentation:
					entity_idx_list = [] # we use list here because each avatar consists of two entities
					for idx, info in enumerate(self.obs["gt_seg_entity_idx_to_info"]):
						if f"{info['type']}:{info['name']}" == focused_event["curr_event"].event_keywords[0]:
							# print(f"{info['type']}:{info['name']}")
							entity_idx_list.append(idx)
					assert len(entity_idx_list) > 0
					# res = self.obs['rgb'].shape[:2]
					# print(res)
					# f_x = res[0] / (2.0 * np.tan(np.radians(self.obs['fov'] / 2.0)))
					# f_y = res[1] / (2.0 * np.tan(np.radians(self.obs['fov'] / 2.0)))
					# intrinsic_matrix = np.array([[f_x, 0.0, res[0]/2.0],
					# 							 [0.0, f_y, res[1]/2.0],
					# 							 [0.0, 0.0, 1.0]])
					# target_position = self.get_3d_bounds_center(entity_idx_list, self.obs['segmentation'], self.obs['depth'], intrinsic_matrix)
					# print(target_position)
					# print(self.pose[:3])
					# distance_between_two_agents = np.linalg.norm(target_position - np.array(self.pose[:3]))
					# print("dist between:", distance_between_two_agents)
					# exit()
					mask_of_obj = np.isin(self.obs["segmentation"], entity_idx_list)
					# print(self.self.obs["depth"].tolist())
					if mask_of_obj.any():
						obj_coords = np.argwhere(mask_of_obj)
						centroid = np.mean(obj_coords, axis=0)
						cy, cx = centroid.astype(int)
						center_depth = self.obs["depth"][cy, cx]
						# print("center_depth:", center_depth)
						# depth_of_obj = self.obs["depth"][mask_of_obj]
						# median_val = np.median(depth_of_obj)
						# abs_deviation = np.abs(depth_of_obj - median_val)
						# mad = np.median(abs_deviation)
						# outlier_threshold = 3 * mad
						# inliers = depth_of_obj[abs_deviation <= outlier_threshold]
						# avg_depth_of_obj = inliers.mean()
					converse_range = center_depth + 3
					# print(utt)
					# print(f"{self.name} generate a converse event with range {converse_range}")
				else:
					target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
					target_position = None
					for cur_obj in self.cur_objects:
						if cur_obj["name"] == target_agent_name:
							print(f"target agent {cur_obj['name']} DISTANCE:", np.linalg.norm(np.array(cur_obj["position"]) - np.array(self.pose[:3])))
							target_position = cur_obj["position"]
							break
					converse_range = np.linalg.norm(np.array(target_position) - np.array(self.pose[:3])) + 1
					print("converse range:", converse_range)
				action = {
					'type': 'converse',
					'arg1': utt,
					'arg2': converse_range
				}
				print(f"{self.name}: {utt}")
				# print('-' * 50)

				return action
		elif focused_event["curr_event"].event_type == "store goods":
			item = self.react_buy_good(focused_event)
			if item is not None:
				if self.held_objects[0] is None:
					action = {
						'type': 'pick',
						'arg1': 0,
						'arg2': item
					}
					self.update_action(action)
					desc = f"Pick up {item} at {self.current_place}."
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.e_mem.add_memory(event_type="action", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.current_place, event_keywords=[item, "pick", ""], event_img=None, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc))
					return action
				elif self.held_objects[1] is None:
					action = {
						'type': 'pick',
						'arg1': 1,
						'arg2': item
					}
					self.update_action(action)
					desc = f"Pick up {item} at {self.current_place}."
					if desc not in self.e_mem.experience_embedding.keys():
						embedding = self.generator_embedding.get_embedding(desc, caller="embed_speech_description")
					else:
						embedding = self.e_mem.experience_embedding[desc]
					self.e_mem.add_memory(event_type="action", event_time=self.curr_time, event_position=self.pose[:3], event_place=self.current_place, event_keywords=[item, "pick", ""], event_img=None, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc))
					return action
				else:
					self.logger.warning(f"I already held two objects {self.held_objects}, cannot pick up {item}.")

		return "continue"
			
	def generate_new_decomp_schedule(self, react_summary):
		hour_str = [
			"00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM",
			"05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM",
			"10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM",
			"03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
			"08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"
		]
		
		def parse_time(s):
			hhmm, ap = s.split()
			hh, mm = map(int, hhmm.split(':'))
			if ap == 'AM' and hh == 12:
				hh = 0
			if ap == 'PM' and hh != 12:
				hh += 12
			return time(hh, mm)
		
		tlist = [parse_time(s) for s in hour_str]
		next_hour_idx = 0
		for i in range(len(tlist) - 1):
			if tlist[i] <= self.curr_time.time() < tlist[i+1]:
				next_hour_idx = i + 1
				break
		
		old_compressed = self.hourly_schedule
		old_hourly_tasks = []
		for task, duration in old_compressed:
			hours_for_task = duration // 60
			old_hourly_tasks.extend([task] * hours_for_task)
		
		new_hourly_tasks = []
		with open("generative_agents/persona/prompt_template/v2/generate_hourly_schedule_v2.txt", "r") as f:
			prompt_template = f.read()
		
		schedule_format = ""
		for h in hour_str:
			schedule_format += f"[{self.get_curr_date()} -- {h}] Activity: [Fill in]\n"
		schedule_format = schedule_format.strip()
		
		prior_schedule = ""
		if next_hour_idx > 0:
			prior_schedule = "\n"
			for count in range(next_hour_idx):
				prior_schedule += f" {self.get_curr_date()} -- {hour_str[count]}] Activity: {self.scratch['first_name']} is {old_hourly_tasks[count]}\n"
		
		intermission_str = f"Here is today's originally intended plan for {self.scratch['first_name']}: "
		for count, p in enumerate(self.daily_plan):
			intermission_str += f"{count+1}) {p}, "
		intermission_str = intermission_str.rstrip(", ")
		intermission_str += f". New info: {react_summary}"
		
		for h_idx in range(next_hour_idx, 24):
			curr_hour_str = hour_str[h_idx]
			prompt = prompt_template
			prompt = prompt.replace("!<INPUT 0>!", schedule_format)
			prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
			prompt = prompt.replace("!<INPUT 2>!", prior_schedule)
			prompt = prompt.replace("!<INPUT 3>!", intermission_str)
			prompt = prompt.replace("!<INPUT 4>!", "")
			prompt = prompt.replace("!<INPUT 5>!", f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is")
			if "<commentblockmarker>###</commentblockmarker>" in prompt:
				prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
			response = self.generator.generate(prompt, caller="generate_new_decomp_schedule")
			cr = response.strip()
			try:
				if '[' in cr:
					cr = cr.split("[")[0].strip()
				if cr.endswith('.'):
					cr = cr[:-1]
			except:
				cr = "sleeping"
			new_hourly_tasks.append(cr)
			prior_schedule += f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is {cr}\n"
		
		combined_hourly_tasks = old_hourly_tasks[:next_hour_idx] + new_hourly_tasks
		_compressed = []
		prev_task = None
		for i in range(len(combined_hourly_tasks)):
			combined_hourly_tasks[i] = combined_hourly_tasks[i].split('\n')[0] if '\n' in combined_hourly_tasks[i] else combined_hourly_tasks[i]
		
		for i in combined_hourly_tasks:
			if i != prev_task:
				_compressed.append([i, 1])
				prev_task = i
			else:
				_compressed[-1][1] += 1
		final_compressed = []
		for task, hour_count in _compressed:
			final_compressed.append([task, hour_count * 60])
		return final_compressed
	
	def _act(self, obs):

		if self.current_place == self.scratch["groups"][0]["place"]:
			if self.held_objects[0] is not None:
				action = {
					'type': 'put',
					'arg1': 0
				}
				self.update_action(action)
				self.logger.info(f"{self.name} is at the group place, put down the object in hand 0.")
				return action
			elif self.held_objects[1] is not None:
				action = {
					'type': 'put',
					'arg1': 1
				}
				self.update_action(action)
				self.logger.info(f"{self.name} is at the group place, put down the object in hand 1.")
				return action

		# print(self.curr_time)
		# print("new schedule:", self.generate_new_decomp_schedule("catching up over dinner with Brian Carter at the new restaurant downtown tomorrow evening."))
		# exit()
		# print("current_place:", obs['current_place'])
		# sleep mechanism (if agent enters into a building, then react once, if not needed, then return sleep)
		# print("obs['curr_time'] != self.curr_goal_end_time:", obs['curr_time'] != self.curr_goal_end_time)

		if self.last_actions[-1] is not None and self.last_actions[-1]["type"] == "enter":
			self.need_react_in_building = True
		if obs["current_place"] is not None or self.commuting == False: # that means the agent is in a building
			if self.curr_goal_address and (obs['curr_time'] != self.curr_goal_end_time):
				has_speech_event_in_building = False
				for event in self.obs["events"]:
					if event["type"] == "speech" and event["position"][:2] != self.pose[:2]:
						has_speech_event_in_building = True
				if (not has_speech_event_in_building) and (not self.need_react_in_building):
					# print(self.curr_time)
					# print(f"{self.name} is sleeping")
					return {
						'type': 'sleep'
					}
				else:
					self.last_react_time = self.curr_time

		# Do memory retrieval first
		retrieved = self.e_mem.retrieve_events_thoughts_by_keywords(self.perceived)
		focused_event = self.get_focused_event(retrieved) # this is used for react() only
		
		if not self.curr_goal_address or obs['curr_time'] == self.curr_goal_end_time:
			self.plan(obs["rgb"])

		if self.reflection_trigger():
			self.reflect()
			self.reset_reflection_counter()
			
		action = self.last_actions_handler()
		if action != "continue":
			# print(f"{self.name} last action handler:", action)
			self.update_action(action)
			return action

		action = self.obs_events_handler()
		if action != "continue":
			# print(f"{self.name} obs events handler:", action)
			self.update_action(action)
			return action
		
		# print(f"{self.name} focused_event is None?:", focused_event == None)
		if self.last_react_time == self.curr_time:
			self.need_react_in_building = False
			if focused_event:
				action = self.focused_event_handler(focused_event)
				if action != "continue":
					self.update_action(action)
					return action
				
		# print(f"{self.name} critical")
		# already at the correct place
		if self.curr_goal_address == obs['current_place'] or self.commuting == False:
			self.logger.info(f"{self.name} at {self.curr_goal_address} is {self.scratch['act_description']} for {self.scratch['act_duration']} minutes.")
			self.update_action(None)
			return None
		# can enter the correct place
		if self.curr_goal_address in obs['accessible_places']:
			self.logger.info(f"{self.name} finished navigation to {self.curr_goal_address} at {self.curr_goal_pos}, begin to {self.curr_goal_description} for {self.curr_goal_duration} minutes.")
			action = {
				'type': 'enter',
				'arg1': self.curr_goal_address
			}
			self.update_action(action)
			self.commuting = False
			return action
		# at wrong place, need to enter open space
		if obs['current_place'] is not None:
			self.logger.info(f"{self.name} at {obs['current_place']} is entering open space to move to {self.curr_goal_address} at {self.curr_goal_pos}.")
			action = {
				'type': 'enter',
				'arg1': 'open space'
			}
			self.update_action(action)
			self.commuting = True
			return action
		# at open space, need to move to the correct place
		cur_trans = np.array(self.pose[:2])
		if is_near_goal(cur_trans[0], cur_trans[1], self.curr_goal_bbox, self.curr_goal_pos):
			self.logger.warning(f"{self.name} at {self.pose[:3]} is near the goal {self.curr_goal_pos}, but not at the goal {self.curr_goal_address}.")
			self.update_action(None)
			return None
		self.logger.info(f"{self.name} at {self.pose[:3]} is moving to {self.curr_goal_address} at {self.curr_goal_pos} to {self.curr_goal_description} for {self.curr_goal_duration} minutes.")
		nav_action = self.navigate()
		self.update_action(nav_action)
		return nav_action

	def navigate(self):
		return super().navigate(self.s_mem.get_sg(self.current_place), self.curr_goal_pos, self.curr_goal_bbox)

	def plan(self, rgb):
		next_schedule_index = self.get_next_schedule_index()
		# print(f"{self.name} next_schedule_index:", next_schedule_index)
		# print("next_schedule_index:", next_schedule_index)
		self.curr_goal_description, self.curr_goal_duration = self.hourly_schedule[next_schedule_index]
		# print(self.curr_goal_description, self.curr_goal_duration)
		self.curr_goal_address = self.generate_goal_address()
		self.commuting = True
		# print("curr_goal_address:", self.curr_goal_address)
		# generative agents:
		# 1. action_sector -> action_arena (e.g. bedroom 2) -> action_game_object (e.g. bed) -> action_pronunciatio (e.g. emoji of sleeping) -> action_event_triple (e.g. (Joon Park, brew, coffee)) -> generate new object states that influenced by persona's actions (including act_obj_desc, action pronunciatio of act_obj_desc, and act_obj_event_triple)
		# print("self.curr_goal_address:", self.curr_goal_address)
		# print("curr goal address:", self.curr_goal_address)
		curr_goal_dict = self.s_mem.get_knowledge(self.curr_goal_address)
		# print("curr_goal_dict:", curr_goal_dict)
		self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]])
		self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"])
		self.scratch["act_address"] = self.curr_goal_address
		self.scratch["act_description"] = self.curr_goal_description
		self.scratch["act_duration"] = self.curr_goal_duration
		curr_goal_start_time = datetime.combine(self.curr_time.date(), datetime.min.time()) + timedelta(minutes=sum(activity[1] for activity in self.hourly_schedule[:next_schedule_index]))
		self.scratch["act_start_time"] = curr_goal_start_time.strftime("%B %d, %Y, %H:%M:%S")
		self.curr_goal_end_time = curr_goal_start_time + timedelta(minutes=self.curr_goal_duration)
		print(f"{self.name} {self.curr_goal_description} end at {self.curr_goal_end_time}")
		self.logger.info(f"{self.name} planned to {self.curr_goal_description} at {self.curr_goal_address} for {self.curr_goal_duration} minutes.")
	
	def react_buy_good(self, focused_event):
		context = ""
		curr_descs = []
		for event in focused_event["events"]:
			curr_descs.append(event["event_description"])
		curr_descs = set(curr_descs)
		for curr_desc in curr_descs:
			context +=  f"{curr_desc}. "
		context += f"\nHere is a brief description of {self.name}:\n{self.get_character_description()}\n"
		focal_points = focused_event["curr_event"].event_keywords
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 50)
		retrieved_strs = []
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_strs.append(event["event_description"])
		retrieved_strs = list(set(retrieved_strs))
		print("retrieved_str:", "\n".join(retrieved_strs))
		return self.generate_decide_to_buy(focused_event, context, "\n".join(retrieved_strs))

	def generate_decide_to_buy(self, focused_event, context, retrieved_str):
		prompt_part1 = f"Context for the task:\nPART 1. {context} Here is the memory that is in {self.name}'s head: {retrieved_str}\n"
		prompt_part2 = f"Task: I am in a store. {focused_event['curr_event'].event_description}. Given the above, only return the name of the item you want to buy. If you do not want to buy anything, return 'nothing'. Do not provide any explanation, reasoning, or additional text."
		prompt = prompt_part1 + "---\n" + prompt_part2
		print("generate_decide_to_buy prompt:", prompt)
		response = self.generator.generate(prompt, caller="generate_decide_to_buy")
		print("generate_decide_to_buy response:", response)
		response = response.strip()
		for good in focused_event['curr_event'].event_keywords:
			if good in response:
				return good
		return None

	def react_converse(self, obs, focused_event, target_knowledge): # only support conversation currently
		def converse_end_procedure():
			target_agent_name = focused_event["curr_event"].event_keywords[0]
			if "avatar:" in target_agent_name:
				target_agent_name = target_agent_name.split("avatar:")[1]
			kws = [target_agent_name, "chat", self.name]
			desc = self.generate_conversation_summarization(self.e_mem.curr_chat)
			# print("generate_conversation_summarization desc:", desc)
			if desc not in self.e_mem.experience_embedding.keys():
				if desc != "":
					embedding = self.generator_embedding.get_embedding(desc, caller="embed_chat")
				else:
					embedding = None
			else:
				embedding = self.e_mem.experience_embedding[desc]
			if embedding is not None:
				self.e_mem.add_memory(event_type="chat", event_time=self.curr_time, event_position=self.pose[:3], event_place=obs['current_place'], event_keywords=kws, event_img=None, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc))
			# print("last chat:", self.e_mem.get_last_chat(target_agent_name))
			# exit()
			if self.last_new_schedule_time is None or (self.curr_time - self.last_new_schedule_time).total_seconds() > self.generate_new_schedule_freq:
				print("conversation summary:", desc)
				try:
					self.hourly_schedule = self.generate_new_decomp_schedule(desc)
					print(self.hourly_schedule)
					self.scratch["hourly_schedule"] = self.hourly_schedule
				except Exception as e:
					print("Error in generating new schedule: {e}")
				self.last_new_schedule_time = self.curr_time
		# if not (target_agent.curr_goal_address and target_agent.curr_goal_description and self.curr_goal_address and self.curr_goal_description):
		# 	return False
		# if "sleep" in target_agent.curr_goal_description or "sleep" in self.curr_goal_description:
		# 	return False
		# if self.curr_time.hour == 23:
		# 	return False
		# if "waiting" in target_agent.curr_goal_description:
		# 	return False
		# print(f"length of curr_chat for {self.name}:", sum(1 for name, _ in self.e_mem.curr_chat if name == self.name))
		print(f"{self.name} curr_chat:", self.e_mem.curr_chat)
		# print(f"{self.name} convo num:", sum(1 for name, _ in self.e_mem.curr_chat if name == self.name))
		# print(f"{self.name} curr_chat_end:", self.e_mem.curr_chat_end)
		if sum(1 for name, _ in self.e_mem.curr_chat if name == self.name) >= 8: # allow max. 8 utterances from any person
			converse_end_procedure()
			self.e_mem.curr_chat_end = False
			self.e_mem.curr_chat = []
			return False
		# if len(self.e_mem.curr_chat) > 1 and self.e_mem.curr_chat[-1][0] == self.name:
		# 	self.e_mem.curr_chat_end = True
		if self.e_mem.curr_chat_end:
			converse_end_procedure()
			self.e_mem.curr_chat_end = False # reset
			self.e_mem.curr_chat = [] # reset
			return False
		return self.generate_decide_to_talk(obs, focused_event, target_knowledge)
		# return True
	
	def generate_event_triple_from_rgb(self, rgb):
		prompt = "Task: Turn the input image into (subject, predicate, object). \nOutput examples: (Woman, eat, breakfast), (Man, brew, coffee), (Child, is, sleep) \nOutput:"
		response = self.generator.generate(prompt, img=rgb, caller="generate_event_triple_from_rgb")
		subject, predicate, object = response.strip("()").split(", ")
		# print("debug:generate_event_triple_from_rgb:gpt_response into spo:", [subject, predicate, object])
		return subject, predicate, object
	
	def generate_event_triple_from_text(self, text):
		prompt = f"Task: Turn the input text '{text}' into (subject, predicate, object). \nOutput examples: (Woman, eat, breakfast), (Man, brew, coffee), (Child, is, sleep) \nOutput:"
		response = self.generator.generate(prompt, caller="generate_event_triple_from_text")
		subject, predicate, object = response.strip("()").split(", ")
		# print("debug:generate_event_triple_from_text:gpt_response into spo:", [subject, predicate, object])
		return subject, predicate, object
	
	def generate_action_event_triple(self, action_description):
		with open ("generative_agents/persona/prompt_template/v2/generate_event_triple_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.name)
		prompt = prompt.replace("!<INPUT 1>!", action_description)
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_action_event_triple")
		try:
			cr = response.strip()
			cr = [i.strip() for i in cr.split(")")[0].split(",")]
			assert len(cr) == 2
			cr = (self.name, cr[0], cr[1])
		except:
			cr = (self.name, "is", "idle")
		return cr
	
	def generate_to_turn(self, speech): 
		prompt = f"My name is {self.name} and I hear someone is talking: {speech} \nShould I turn around to the source of the sound? Respond with only 'yes' or 'no'."
		response = self.generator.generate(prompt, caller="generate_to_turn")
		if "yes" in response.lower() and "no" not in response.lower():
			return True
		return False

	def describe_knowledge(self, knowledge):
		if knowledge is None:
			return "No knowledge."
		desc = ""
		for key, value in knowledge.items():
			if ".png" in str(value):
				continue
			if "_idx" in key:
				continue
			desc += f"{key}: {value}\n"
		desc = f"{{\n{desc}}}\n"
		return desc

	def generate_decide_to_talk(self, obs, focused_event, target_knowledge):
		target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
		# print(f"target agent name is {target_agent_name}")
		last_chat = self.e_mem.get_last_chat(target_agent_name)
		if last_chat: 
			last_chat_time = last_chat["time"]
			last_chat_desciption = last_chat["description"]
		else:
			last_chat_time = ""
			last_chat_desciption = ""
		context = ""
		curr_descs = []
		for event in focused_event["events"]:
			curr_descs.append(event["event_description"]) # .split(" ")
			# curr_desc[2:3] = ["was"]
			# curr_desc = " ".join(curr_desc)
		curr_descs = set(curr_descs)
		for curr_desc in curr_descs:
			context +=  f"{curr_desc}. "
		# print("debug:generate_decide_to_talk:content:", context)
		# for event in focused_event["events"]: 
		# 	curr_desc = event["event_keywords"]
		# 	curr_desc = " ".join(curr_desc)
		# 	context +=  f"{curr_desc} in the past."
		context += f"\nHere is a brief description of {self.name}:\n{self.get_character_description()}\n"
		context += f"Knowledge about {target_agent_name}:\n{self.describe_knowledge(target_knowledge)}"
		curr_time = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		if self.curr_goal_address == obs['current_place'] and "waiting" not in self.curr_goal_description: 
			curr_act_description = f"{self.name} is already {self.curr_goal_description}"
		elif "waiting" in self.curr_goal_description:
			curr_act_description = f"{self.name} is {self.curr_goal_description}"
		else: 
			curr_act_description = f"{self.name} is on the way to {self.curr_goal_description}"
		target_curr_act_description = ""
		for event in self.obs["events"]:
			if event["type"] == "speech" and event["content"]["from_name"] == target_agent_name:
				target_curr_act_description = f"{target_agent_name} is talking: {event['content']['utterance']}"
		with open ("generative_agents/persona/prompt_template/v2/decide_to_talk_v2.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", context)
		prompt = prompt.replace("!<INPUT 1>!", curr_time)
		prompt = prompt.replace("!<INPUT 2>!", self.name)
		prompt = prompt.replace("!<INPUT 3>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 4>!", last_chat_time)
		prompt = prompt.replace("!<INPUT 5>!", last_chat_desciption)
		prompt = prompt.replace("!<INPUT 6>!", curr_act_description)
		prompt = prompt.replace("!<INPUT 7>!", target_curr_act_description)
		prompt = prompt.replace("!<INPUT 8>!", self.name)
		prompt = prompt.replace("!<INPUT 9>!", target_agent_name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_decide_to_talk")
		# print("debug::generate_decide_to_talk::prompt:", prompt)
		# print("debug::generate_decide_to_talk::gpt_response:", response)
		# if os.path.exists('debug:decide_to_talk.json'):
		# 	with open('debug:decide_to_talk.json', 'r') as json_file:
		# 		existing_prints = json.load(json_file)
		# else:
		# 	existing_prints = []
		# existing_prints.append({
		# 	"prompt": prompt,
		# 	"output": response,
		# })
		# with open('debug:decide_to_talk.json', 'w') as json_file:
		# 	json.dump(existing_prints, json_file, indent=4)
		if "yes" in response.lower():
			self.logger.debug("{self.name}@generate_decide_to_talk:True")
			return True
		self.logger.debug("{self.name}@generate_decide_to_talk:False")
		return False
	
	def converse(self, obs, focused_event, target_agent_state=None):
		target_agent_name = focused_event["curr_event"].event_keywords[0].split('avatar:')[1]
		focal_points = [target_agent_name]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 50)
		retrieved_str = ""
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_str += f"{event['event_description']}\n"
		# chat_history_with_target_agent = self.e_mem.get_all_chat(target_agent_name, max=50)
		# chat_history_with_target_agent_verbalize = ""
		# if chat_history_with_target_agent is not None:
		# 	for chat in chat_history_with_target_agent:
		# 		chat_verbalize = chat["time"].strftime("%B %d, %Y, %H:%M:%S") + ": " + chat["description"]
		# 		chat_history_with_target_agent_verbalize += "Chat history at " + chat_verbalize + '\n'
		# statements = chat_history_with_target_agent_verbalize
		statements = retrieved_str
		# target_agent_in_groups = False
		for group in self.scratch["groups"]:
			if target_agent_name in group["members"]:
				statements += f'{target_agent_name} is also in the {group["name"]} which is {group["description"][0].lower() + group["description"][1:]} located at the {group["place"]}.\n'
			else:
				statements += f"{self.name} and {target_agent_name} don't have group relationships"
		if statements == "":
			statements = f"{self.name} and {target_agent_name} don't know each other."
		relationship = self.generate_chat_summarize_relationship(target_agent_name, statements)
		last_chat = ""
		for chat_single in self.e_mem.curr_chat[-4:]:
			last_chat += ": ".join(chat_single) + "\n"
		if last_chat:
			focal_points = [f"{relationship}", 
                      f"{target_agent_name} is {target_agent_state}" if target_agent_state is not None else f"{target_agent_name}", 
                      last_chat]
		else:
			focal_points = [f"{relationship}", 
                      f"{target_agent_name} is {target_agent_state}" if target_agent_state is not None else f"{target_agent_name}"]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 15)
		# chat_history_with_target_agent = self.e_mem.get_all_chat(target_agent_name, max=15)
		# chat_history_with_target_agent_verbalize = ""
		# if chat_history_with_target_agent is not None:
		# 	for chat in chat_history_with_target_agent:
		# 		chat_verbalize = chat["time"].strftime("%B %d, %Y, %H:%M:%S") + ": " + chat["description"]
		# 		chat_history_with_target_agent_verbalize += chat_verbalize + '\n'
		# print("chat history:", chat_history_with_target_agent_verbalize)
		current_place = obs["current_place"] if obs["current_place"] is not None else "outdoor"
		# print(f"{self.name}: curr_chat:")
		# for message in self.e_mem.curr_chat:
		# 	print(f"{message[0]}: {message[1]}")
		utt, end = self.generate_one_utterance(target_agent_name, target_agent_state, current_place, relationship, retrieved, self.e_mem.curr_chat)
		# print(f"{self.name}: utterance:", utt)
		# print(f"{self.name}: end:", end)
		# print('-' * 25)
		self.e_mem.curr_chat += [[self.name, utt]]
		if end:
			self.e_mem.curr_chat_end = True
			# self.e_mem.add_chat(self.curr_time, target_agent_name, description=self.generate_conversation_summarization(self.e_mem.curr_chat))
			kws = ["avatar" + ':' + target_agent_name, "chat", "avatar" + ':' + self.name]
			desc = self.generate_conversation_summarization(self.e_mem.curr_chat)
			if desc not in self.e_mem.experience_embedding.keys():
				if desc != "":
					embedding = self.generator_embedding.get_embedding(desc, caller="embed_chat")
				else:
					embedding = None
			else:
				embedding = self.e_mem.experience_embedding[desc]
			if embedding is not None:
				self.e_mem.add_memory(event_type="chat", event_time=self.curr_time, event_position=self.pose[:3], event_place=obs['current_place'], event_keywords=kws, event_img=None, event_description=desc, event_text_ft=embedding, event_poignancy=self.generate_poig_score(desc))
		# self.curr_goal_address = target_agent_name # walk to target agent
		# self.curr_goal_pos = np.array([curr_goal_dict["location"][0], curr_goal_dict["location"][1]]) ???
		# self.curr_goal_bbox = np.array(curr_goal_dict["bounding_box"]) ???
		# self.curr_goal_description = f"Conversation with {target_agent_name}"
		# self.curr_goal_duration = # how to use second because this is minute based, use 1/60?
		return utt, end
	
	def extract_first_json_dict(self, data_str):
		start_idx = data_str.find('{')
		end_idx = data_str.find('}', start_idx) + 1
		if start_idx == -1 or end_idx == 0:
			return None
		json_str = data_str[start_idx:end_idx]
		try:
			json_dict = json.loads(json_str)
			return json_dict
		except json.JSONDecodeError:
			return None
		
	def human_readable_time_difference(self, current_time, past_time):
		time_diff = current_time - past_time
		total_seconds = int(time_diff.total_seconds())
		days = total_seconds // 86400
		hours = (total_seconds % 86400) // 3600
		minutes = (total_seconds % 3600) // 60
		seconds = total_seconds % 60
		result = []
		if days > 0:
			result.append(f"{days} day{'s' if days > 1 else ''}")
		if hours > 0:
			result.append(f"{hours} hour{'s' if hours > 1 else ''}")
		if minutes > 0:
			result.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
		if seconds > 0:
			result.append(f"{seconds} second{'s' if seconds > 1 else ''}")
		return " ".join(result)

	def generate_one_utterance(self, target_agent_name, target_agent_state, current_place, relationship, retrieved, curr_chat):
		target_agent_state_ing_form_map = {"walking": "walking", "bicycle": "bicycling", "bus": "riding the bus", "null": None}
		retrieved_strs = []
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				retrieved_strs.append(f"{event['event_description']}")
		retrieved_strs = set(retrieved_strs)
		retrieved_str = ""
		for retrieved_str in retrieved_strs:
			retrieved_str += f"{retrieved_str}.\n"
		curr_context = (f"{self.name} " + 
              f"was {self.curr_goal_description} " + 
              f"when {self.name} " + 
              f"saw {target_agent_name}" )
		if target_agent_state != None:
			curr_context += f" in the middle of {target_agent_state_ing_form_map[target_agent_state]}.\n"
		else:
			curr_context += ".\n"
		curr_context += f"The relationship between {self.name} and {target_agent_name}: {relationship}\n"
		curr_context += (f"{self.name} " +
              f"is initiating a conversation with " +
              f"{target_agent_name}.")
		prev_conversation_insert = "\n"
		last_chat = self.e_mem.get_last_chat(target_agent_name)
		if last_chat is not None:
			if int((self.curr_time - datetime.strptime(last_chat["time"], "%B %d, %Y, %H:%M:%S")).total_seconds() / 60) > 480: 
				prev_conversation_insert = ""
			else:
				prev_conversation_insert += f"{self.human_readable_time_difference(current_time=self.curr_time, past_time=datetime.strptime(last_chat['time'], '%B %d, %Y, %H:%M:%S'))} minutes ago, {self.name} and {target_agent_name} were already {last_chat['description']}. This context takes place after that conversation."
		if prev_conversation_insert == "\n": 
			prev_conversation_insert = ""
		curr_chat_verbalize = ""
		for chat_single in curr_chat:
			curr_chat_verbalize += ": ".join(chat_single) + "\n"
		if curr_chat_verbalize == "":
			curr_chat_verbalize = "[The conversation has not started yet -- start it!]"
		# print("curr_chat_verbalize:", curr_chat_verbalize)
		intro_character_description = f"Here is a brief description of {self.name}.\n{self.get_character_description()}"
		with open ("generative_agents/persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", intro_character_description)
		prompt = prompt.replace("!<INPUT 1>!", self.name)
		prompt = prompt.replace("!<INPUT 2>!", retrieved_str)
		prompt = prompt.replace("!<INPUT 3>!", prev_conversation_insert)
		prompt = prompt.replace("!<INPUT 4>!", current_place)
		prompt = prompt.replace("!<INPUT 5>!", curr_context)
		prompt = prompt.replace("!<INPUT 6>!", self.name)
		prompt = prompt.replace("!<INPUT 7>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 8>!", curr_chat_verbalize)
		prompt = prompt.replace("!<INPUT 9>!", self.name)
		prompt = prompt.replace("!<INPUT 10>!", target_agent_name)
		prompt = prompt.replace("!<INPUT 11>!", self.name)
		prompt = prompt.replace("!<INPUT 12>!", self.name)
		prompt = prompt.replace("!<INPUT 13>!", self.name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_one_utterance")
		# print("debug::generate_one_utterance::prompt:", prompt)
		# print("debug::generate_one_utterance::gpt_response:", response)
		# if os.path.exists('debug:generate_one_utterance.json'):
		# 	with open('debug:generate_one_utterance.json', 'r') as json_file:
		# 		existing_prints = json.load(json_file)
		# else:
		# 	existing_prints = []
		# existing_prints.append({
		# 	"prompt": prompt,
		# 	"output": response,
		# })
		# with open('debug:generate_one_utterance.json', 'w') as json_file:
		# 	json.dump(existing_prints, json_file, indent=4)
		try:
			gpt_response_dict = self.extract_first_json_dict(response)
			cleaned_dict = dict()
			cleaned = []
			for key, val in gpt_response_dict.items(): 
				cleaned += [val]
				cleaned_dict["utterance"] = cleaned[0]
				cleaned_dict["end"] = True
			if "f" in str(cleaned[1]) or "F" in str(cleaned[1]): 
				cleaned_dict["end"] = False
		except:
			cleaned_dict = dict()
			cleaned_dict["utterance"] = "..."
			cleaned_dict["end"] = False
		return cleaned_dict["utterance"], cleaned_dict["end"]

	def generate_chat_summarize_relationship(self, target_persona_name, statements):
		# print("debug::generate_chat_summarize_relationship::statements:", statements)
		with open ("generative_agents/persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", statements)
		prompt = prompt.replace("!<INPUT 1>!", self.name)
		prompt = prompt.replace("!<INPUT 2>!", target_persona_name)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_chat_summarize_relationship")
		# print("debug::generate_chat_summarize_relationship::prompt:", prompt)
		# print("debug::generate_chat_summarize_relationship::gpt_response:", response)
		return response
	
	def generate_conversation_summarization(self, curr_chat):
		conversation_str = ""
		for row in curr_chat: 
			conversation_str += f'{row[0]}: "{row[1]}"\n'
		with open ("generative_agents/persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", conversation_str)
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		example_output = "conversing about eating dinner with Brian Carter in Boston Sail Loft at 18:00."
		special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who, when, where if applicable)."
		response = self.generator.generate(self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction), caller="generate_conversation_summarization")
		# print("debug::generate_conversation_summarization::prompt:", self.compose_instruction_prompts(prompt=prompt, example_output=example_output, special_instruction=special_instruction))
		# print("debug::generate_conversation_summarization::gpt_response:", response)
		return response
	
	def generate_goal_address(self):
		assert self.scratch["living_place"] is not None
		places_in_spatial_memory = self.s_mem.get_places()
		places_in_spatial_memory_formatted = [f'"{place}"' for place in
												 places_in_spatial_memory]
		prompt = f"My living place is {self.scratch['living_place']}. Currently, I want to do this action: {self.curr_goal_description}, what is the most possible place for doing this action in the following list of places: [{','.join(places_in_spatial_memory_formatted)}]. Choose only one place from the list and output it as the only response. Do not provide any explanation, reasoning, or additional text."
		
		response = self.generator.generate(prompt, caller="generate_goal_address")
		# print("debug::generate_goal_address:response:", response)
		cr = self.scratch['living_place']
		try:
			for place in places_in_spatial_memory:
				if place in response:
					cr = place
					break
		except:
			cr = self.scratch['living_place']
		return cr
	
	def revise_identity_and_generate_daily_plan(self):
		self.wake_up_hour = self.generate_wake_up_hour()
		focal_points = [f"{self.name}'s plan for {self.get_curr_date()}.",
						f"Important recent events for {self.name}'s life."]
		focal_points_embedding = []
		for focal_point in focal_points:
			if focal_point != "":
				focal_points_embedding.append(self.generator_embedding.get_embedding(focal_point, caller="embed_focal_point") if focal_point not in self.e_mem.experience_embedding.keys() else self.e_mem.experience_embedding[focal_point])
		retrieved = self.e_mem.new_retrieve(self.curr_time, focal_points, focal_points_embedding, 30)
		statements = "[Statements]\n"
		for focal_point_retrieved in retrieved:
			for event in retrieved[focal_point_retrieved]:
				statements += f"{event['event_time']}: {event['event_description']}\n"

		plan_prompt = statements + "\n"
		plan_prompt += f"Given the statements above, is there anything that {self.name} should remember as they plan for"
		plan_prompt += f" *{self.curr_time.strftime('%A %B %d')}*? "
		plan_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement)\n\n"
		plan_prompt += f"Write the response from {self.name}'s perspective."
		plan_note = self.generator.generate(plan_prompt, caller="revise_identity")

		thought_prompt = statements + "\n"
		thought_prompt += f"Given the statements above, how might we summarize {self.name}'s feelings about their days up to now?\n\n"
		thought_prompt += f"Write the response from {self.name}'s perspective."
		thought_note = self.generator.generate(thought_prompt, caller="revise_identity")

		currently_prompt = f"{self.name}'s status from {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}:\n"
		currently_prompt += f"{self.scratch['currently']}\n\n"
		currently_prompt += f"{self.name}'s thoughts at the end of {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}:\n" 
		currently_prompt += ((plan_note if plan_note is not None else "") + (thought_note if thought_note is not None else "")).replace('\n', '') + "\n\n"
		currently_prompt += f"It is now {self.curr_time.strftime('%A %B %d')}. Given the above, write {self.name}'s status for {self.curr_time.strftime('%A %B %d')} that reflects {self.name}'s thoughts at the end of {(self.curr_time - timedelta(days=1)).strftime('%A %B %d')}. Write this in third-person talking about {self.name}."
		currently_prompt += f"If there is any scheduling information, be as specific as possible (include date, time, and location if stated in the statement).\n\n"
		currently_prompt += "Follow this format below:\nStatus: <new status>"
		new_currently = self.generator.generate(currently_prompt, caller="revise_identity")

		self.scratch['currently'] = new_currently

		daily_req_prompt = self.get_character_description() + "\n"
		daily_req_prompt += f"Today is {self.curr_time.strftime('%A %B %d')}. Here is {self.name}'s plan today in broad-strokes (with the time of the day. e.g., have a lunch at 12:00 pm, watch TV from 7 to 8 pm).\n\n"
		daily_req_prompt += f"Follow this format (the list should have 4~6 items but no more):\n"
		daily_req_prompt += f"1. wake up and complete the morning routine at <time>, 2. ..."

		response = self.generator.generate(daily_req_prompt, caller="revise_identity")
		# print(f"revise_identity generate daily plan:", response)
		# response = response.replace('\n', ' ')
		try:
			lines = response.strip().split('\n')
			cr = [line.split('. ', 1)[1].strip().lower() for line in lines]
		except:
			cr = ['wake up and complete the morning routine at 6:00 am',
				  'eat breakfast at 7:00 am',
				  'read a book from 8:00 am to 12:00 pm',
				  'have lunch at 12:00 pm',
				  'take a nap from 1:00 pm to 4:00 pm',
				  'relax and watch TV from 7:00 pm to 8:00 pm',
				  'finish night routine and sleep at 11:00 pm']
		cr = ([f"wake up and complete the morning routine at {self.wake_up_hour}:00 am"]
              + cr)
		return cr

	def generate_daily_plan(self):
		self.wake_up_hour = self.generate_wake_up_hour()
		with open ("generative_agents/persona/prompt_template/v2/daily_planning_v6.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 1>!", self.scratch['lifestyle'])
		prompt = prompt.replace("!<INPUT 2>!", self.get_curr_date())
		prompt = prompt.replace("!<INPUT 3>!", self.scratch['first_name'])
		prompt = prompt.replace("!<INPUT 4>!", f"{self.wake_up_hour}:00 am")
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		# print("debug:generate_daily_plan:prompt:", f"r'''{prompt}'''")
		response = self.generator.generate(prompt, caller="generate_daily_plan")
		# print("debug:generate_daily_plan:response:", response)
		try:
			cr = []
			_cr = response.split(")")
			for i in _cr:
				if i[-1].isdigit():
					i = i[:-1].strip()
					if i[-1] == "." or i[-1] == ",":
						cr += [i[:-1].strip()]
		except:
			cr = ['wake up and complete the morning routine at 6:00 am',
				  'eat breakfast at 7:00 am',
				  'read a book from 8:00 am to 12:00 pm',
				  'have lunch at 12:00 pm',
				  'take a nap from 1:00 pm to 4:00 pm',
				  'relax and watch TV from 7:00 pm to 8:00 pm',
				  'finish night routine and sleep at 11:00 pm']
		cr = ([f"wake up and complete the morning routine at {self.wake_up_hour}:00 am"]
              + cr)
		return cr

	def generate_hourly_schedule(self):
		hour_str = ["00:00 AM", "01:00 AM", "02:00 AM", "03:00 AM", "04:00 AM",
					"05:00 AM", "06:00 AM", "07:00 AM", "08:00 AM", "09:00 AM",
					"10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM",
					"03:00 PM", "04:00 PM", "05:00 PM", "06:00 PM", "07:00 PM",
					"08:00 PM", "09:00 PM", "10:00 PM", "11:00 PM"]
		n_m1_activity = []
		diversity_repeat_count = 3
		wake_up_hour = self.wake_up_hour
		for i in range(diversity_repeat_count):
			n_m1_activity_set = set(n_m1_activity)
			if len(n_m1_activity_set) < 5:
				n_m1_activity = []
				for count, curr_hour_str in enumerate(hour_str):
					if wake_up_hour > 0:
						n_m1_activity += ["sleeping"]
						wake_up_hour -= 1
					else:
						n_m1_activity += [self.generate_curr_hour_schedule(curr_hour_str, n_m1_activity, hour_str)]

		# Step 1. Compressing the hourly schedule to the following format:
		# The integer indicates the number of hours. They should add up to 24.
		# [['sleeping', 6], ['waking up and starting her morning routine', 1],
		# ['eating breakfast', 1], ['getting ready for the day', 1],
		# ['working on her painting', 2], ['taking a break', 1],
		# ['having lunch', 1], ['working on her painting', 3],
		# ['taking a break', 2], ['working on her painting', 2],
		# ['relaxing and watching TV', 1], ['going to bed', 1], ['sleeping', 2]]
		_n_m1_hourly_compressed = []
		prev = None
		prev_count = 0
		for i in n_m1_activity:
			if i != prev:
				prev_count = 1
				_n_m1_hourly_compressed += [[i, prev_count]]
				prev = i
			else:
				if _n_m1_hourly_compressed:
					_n_m1_hourly_compressed[-1][1] += 1

		# Step 2. Expand to min scale (from hour scale)
		# [['sleeping', 360], ['waking up and starting her morning routine', 60],
		# ['eating breakfast', 60],..
		n_m1_hourly_compressed = []
		for task, duration in _n_m1_hourly_compressed:
			task_cleaned = task
			if '\n' in task_cleaned:
				task_cleaned = task.split('\n')[0]
			n_m1_hourly_compressed += [[task_cleaned, duration * 60]]
		
		# print("n_m1_hourly_compressed:", n_m1_hourly_compressed)

		return n_m1_hourly_compressed

	def generate_curr_hour_schedule(self, curr_hour_str, p_f_ds_hourly_org, hour_str):
		with open ("generative_agents/persona/prompt_template/v2/generate_hourly_schedule_v2.txt", "r") as f:
			prompt_template = f.read()
		schedule_format = ""
		for i in hour_str:
			schedule_format += f"[{self.get_curr_date()} -- {i}]"
			schedule_format += f" Activity: [Fill in]\n"
		schedule_format = schedule_format[:-1]

		prior_schedule = ""
		if p_f_ds_hourly_org:
			prior_schedule = "\n"
			for count, i in enumerate(p_f_ds_hourly_org):
				prior_schedule += f" {self.get_curr_date()} -- {hour_str[count]}] Activity: {self.scratch['first_name']} is {i}\n"

		intermission_str = f"Here the originally intended hourly breakdown of"
		intermission_str += f" {self.scratch['first_name']}'s schedule today: "
		for count, i in enumerate(self.daily_plan):
			intermission_str += f"{str(count + 1)}) {i}, "
		intermission_str = intermission_str[:-2]
		prompt = prompt_template.replace("!<INPUT 0>!", schedule_format)
		prompt = prompt.replace("!<INPUT 1>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 2>!", prior_schedule)
		prompt = prompt.replace("!<INPUT 3>!", intermission_str)
		prompt = prompt.replace("!<INPUT 4>!", "")
		prompt = prompt.replace("!<INPUT 5>!", f" {self.get_curr_date()} -- {curr_hour_str}] Activity: {self.scratch['first_name']} is")
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
		response = self.generator.generate(prompt, caller="generate_curr_hour_schedule")
		try:
			cr = response.strip()
			if '[' in cr:
				cr = cr.split("[")[0].strip()
			if cr[-1] == ".":
				cr = cr[:-1]
		except:
			cr = "sleeping"
		return cr

	def generate_wake_up_hour(self):
		with open ("generative_agents/persona/prompt_template/v2/wake_up_hour_v1.txt", "r") as f:
			prompt_template = f.read()
		prompt = prompt_template.replace("!<INPUT 0>!", self.get_character_description())
		prompt = prompt.replace("!<INPUT 1>!", self.scratch['lifestyle'])
		prompt = prompt.replace("!<INPUT 2>!", self.scratch['first_name'])
		if "<commentblockmarker>###</commentblockmarker>" in prompt:
			prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

		response = self.generator.generate(prompt, caller="generate_wake_up_hour")
		try:
			cr = int(response.strip().lower().split("am")[0])
		except:
			cr = 8
		return cr

	def get_next_schedule_index(self):
		# We first calculate teh number of minutes elapsed today. 
		today_min_elapsed = 0
		today_min_elapsed += self.curr_time.hour * 60
		today_min_elapsed += self.curr_time.minute

		# We then calculate the current index based on that. 
		curr_index = 0
		elapsed = 0
		for task, duration in self.hourly_schedule:
			elapsed += duration
			if elapsed > today_min_elapsed:
				return curr_index
			curr_index += 1

		return curr_index

	def get_curr_date(self):
		if self.curr_time is None:
			return None
		return self.curr_time.strftime("%A %B %d")

	def get_character_description(self):
		"""EXAMPLE OUTPUT
		   Name: Dolores Heitmiller
		   Age: 28
		   Innate traits: hard-edged, independent, loyal
		   Learned traits: Dolores is a painter who wants live quietly and paint
			 while enjoying her everyday life.
		   Currently: Dolores is preparing for her first solo show. She mostly
			 works from home.
		   Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats
			 dinner around 6pm.
		   World Daily Schedule: Dolores is planning to stay at home all day and
			 never go out. [missed now]
		   Current Date: Monday, January 1
		"""
		return f"""Name: {self.name}
			Age: {self.scratch['age']}
			Innate traits: {self.scratch['innate']}
			Learned traits: {self.scratch['learned']}
			Currently: {self.scratch['currently']}
			Lifestyle: {self.scratch['lifestyle']}
			Groups: {self.scratch['groups']}
			Daily plan requirement: {self.scratch['daily_requirement']}
			Held objects: {self.held_objects}
			Current Date: {self.get_curr_date()}
		"""
	
	def compose_instruction_prompts(self, prompt, example_output, special_instruction):
		prompt = '"""\n' + prompt + '\n"""\n'
		prompt += f"Output the response to the prompt above. {special_instruction}\n"
		prompt += "Example output:\n"
		prompt += str(example_output)
		return prompt

if __name__ == '__main__':
	config = json.load(open("ViCo/assets/mit/agents_num_5/config.json", "r"))
	agent = GenAgent(config["agent_names"][1], config["agent_poses"][1], config["agent_infos"][1],
					  "ViCo/assets/mit/ella_agents_num_5", debug=True)
	print(agent.get_character_description())
	print(agent.generate_daily_plan())