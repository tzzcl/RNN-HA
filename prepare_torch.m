imdb = load('train_imdb_Vehi.mat');
raw_dir = '/home/zhangcl/torch_car_images/';
raw_data_dir = '/home/zhangcl/VehicleID_V1.0/image/';
for i = 1 : size(imdb.images.set,2)
    now_name = imdb.images.name{i};
    now_dir = [raw_dir 'train'];
    mkdir([now_dir '/' num2str(imdb.images.class(i))]);
    copyfile([raw_data_dir now_name],[[now_dir '/' num2str(imdb.images.class(i))] '/' now_name]);
    now_val_dir = [raw_dir 'val'];
    if ~exist([now_val_dir '/' num2str(imdb.images.class(i))],'dir')
       mkdir([now_val_dir '/' num2str(imdb.images.class(i))]);
       copyfile([raw_data_dir now_name],[[now_val_dir '/' num2str(imdb.images.class(i))] '/' now_name]);
    end
end
