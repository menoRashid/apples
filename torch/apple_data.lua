do 
	local data = torch.class('data');

	function data:__init(args)
        self.file_path_pos = args.file_path_pos
        self.file_path_neg = args.file_path_neg
        self.limit = args.limit;
        self.mean_file = args.mean_file;
        self.shuffle = args.shuffle;
        self.limit = args.limit;

        
        self.batch_size_pos = args.batch_size_pos;
        self.batch_size_neg = args.batch_size_neg;
        assert (self.batch_size_pos>0);
        assert (self.batch_size_neg>0);
        self.input_size = {240,250,1600};
        self.mean_data=npy4th.loadnpy(self.mean_file);


        self.start_idx_pos = 1;
        self.start_idx_neg = 1;
        -- self.humanImage = false;
        self.training_set = {};
        
        self.lines_pos = self:readDataFile(self.file_path_pos);
        self.lines_neg = self:readDataFile(self.file_path_neg);
        
        if self.shuffle then
        	self.lines_pos = self:shuffleLines(self.lines_pos);
        	self.lines_neg = self:shuffleLines(self.lines_neg);
        end

        if self.limit ~= nil then
            local lines_pos = self.lines_pos;
            self.lines_pos = {};
            local lines_neg = self.lines_neg;
            self.lines_neg = {};
            for i = 1,self.limit do
                self.lines_pos[#self.lines_pos+1] = lines_pos[i];
                self.lines_neg[#self.lines_neg+1] = lines_neg[i];
            end
        end
        print (#self.lines_pos,#self.lines_neg);

	end 

	function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;
        local shuffle = torch.randperm(len)
        local lines={};
        for idx=1,len do
            lines[idx]=x[shuffle[idx]];
        end
        return lines;
    end

    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            -- local start_idx, end_idx = string.find(line, ' ');
            -- local img_label = string.sub(line,1,start_idx-1);

            -- local string_temp = string.sub(line,end_idx+1,#line);
            -- local start_idx, end_idx  =  string.find(string_temp, ' ');
            -- local size_r = string.sub(string_temp,1,start_idx-1);
            -- local size_c = string.sub(string_temp,end_idx+1,#string_temp);
            file_lines[#file_lines+1] =line
             -- {img_label,tonumber(size_r),tonumber(size_c)};

            -- local img_label=string.sub(line,end_idx+1,#line);
            -- file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines
    end

    function data:getTrainingData()
        local total_batch_size=self.batch_size_pos+self.batch_size_neg;
        
        local start_idx_neg_before = self.start_idx_neg
        local start_idx_pos_before = self.start_idx_pos

        self.training_set.data = torch.zeros(total_batch_size,self.input_size[1],
        										self.input_size[2],self.input_size[3]);
        self.training_set.label = torch.zeros(total_batch_size,1);

        self.start_idx_pos = self:addTrainingData(self.training_set,
            self.batch_size_pos,
            self.lines_pos,self.start_idx_pos,0,1)
        
        self.start_idx_neg = self:addTrainingData(self.training_set,
            self.batch_size_neg,
            self.lines_neg,self.start_idx_neg,self.batch_size_pos,-1)

        if self.start_idx_neg<start_idx_neg_before then
            print ('shuffling neg'..self.start_idx_neg..' '..start_idx_neg_before )
            self.lines_neg = self:shuffleLines(self.lines_neg);
        end

        if self.start_idx_pos<start_idx_pos_before then
            print ('shuffling pos'..self.start_idx_pos..' '..start_idx_pos_before )
            self.lines_pos=self:shuffleLines(self.lines_pos);
        end
    end

    function data:addTrainingData(training_set,num_im,list_files,start_idx,start_idx_batch,label)
        local list_idx = start_idx;
        local list_size = #list_files;
        for curr_idx=1,num_im do
            local img_path = list_files[list_idx];
            local img = npy4th.loadnpy(img_path);
            -- print (torch.min(img),torch.max(img));
            -- print (torch.min(self.mean_data),torch.max(self.mean_data),self.mean_data:size());
            img = img:csub(self.mean_data);
            training_set.data[start_idx_batch+curr_idx] = img;
            training_set.label[start_idx_batch+curr_idx] = label;
            list_idx = (list_idx%list_size)+1;
        end
        return list_idx;
    end


end

return data