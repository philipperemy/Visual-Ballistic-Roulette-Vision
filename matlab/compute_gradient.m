clear;clc;
start_idx = 1;
end_idx = 1e9;
prev_img = imread(sprintf('../videos/frames/output_%04d.png', start_idx));
for i = (start_idx+1):end_idx
    try
        img = imread(sprintf('../videos/frames/output_%04d.png', i));
        gradient = img - prev_img;
        % imshow(gradient);
        imwrite(gradient, sprintf('../videos/gradients/output_%04d.png', i));
        prev_img = img;
        fprintf('%d\n', i);
    catch e
        disp(e);
        exit();
    end
end
exit();