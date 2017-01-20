function [ ] = compute_gradient_func( full_video_path )
    % videos/video_dec_24_deutsche_bordeaux/1.mp4
    [~, name, ~] = fileparts(full_video_path);
    fprintf('video_name = %s\n', name);
    % clear;clc;
    start_idx = 1;
    end_idx = 1e9;
    augmented_name = strcat('../output/', name, '/videos');
    if length(ls(strcat(augmented_name, '/gradients'))) < 100
        prev_img = imread(sprintf(strcat(augmented_name, '/frames/output_%04d.png'), start_idx));
        for i = (start_idx+1):end_idx
            try
                img = imread(sprintf(strcat(augmented_name, '/frames/output_%04d.png'), i));
                gradient = img - prev_img;
                % imshow(gradient);
                output_filename = sprintf(strcat(augmented_name, '/gradients/output_%04d.png'), i);
                disp(output_filename);
                imwrite(gradient, output_filename);
                prev_img = img;
                fprintf('%d\n', i);
            catch e
                disp(e);
                exit();
            end
        end
    else
        disp('Nothing to do. Directory seems to be already full.');
    end
    exit();

end

