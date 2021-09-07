
close all;

if ~exist('ld')
    clear all;
    'loading data'
    %d=dir('Stroke hem closest to midline_2019-07-24-12-47-59.bmode.iq.mat');
    
    file_name = '2019-05-31-15-27-10_third injection (3)-2019-05-31-13-54-26_1.bmode.iq.mat';
    load('2019-05-31-15-27-10_third injection (3)-2019-05-31-13-54-26_1.bmode.iq.mat');
    'envelope calculation'
    E=(squeeze(I).^2+squeeze(Q).^2).^0.5;
    ld='true';
    return;
else
    IR=reshape(I,paramIQ.NumSamples*paramIQ.NumLines,paramIQ.NumFrames);
    QR=reshape(Q,paramIQ.NumSamples*paramIQ.NumLines,paramIQ.NumFrames);
    [UI,SI,VI]=svd(IR,'econ');
    [UQ,SQ,VQ]=svd(QR,'econ');
    ThresholdL=200;
    ThresholdH=800;
    SIF=zeros(size(SI));
    SQF=zeros(size(SQ));
    SIF(ThresholdL:ThresholdH,ThresholdL:ThresholdH)=SI(ThresholdL:ThresholdH,ThresholdL:ThresholdH);
    SQF(ThresholdL:ThresholdH,ThresholdL:ThresholdH)=SQ(ThresholdL:ThresholdH,ThresholdL:ThresholdH);
    
    IRF=UI*SIF*VI';
    QRF=UQ*SQF*VQ';
    IF=reshape(IRF,paramIQ.NumSamples,paramIQ.NumLines,paramIQ.NumFrames);
    QF=reshape(QRF,paramIQ.NumSamples,paramIQ.NumLines,paramIQ.NumFrames);
    EF=(IF.^2+QF.^2).^0.5;

    DR=[20 60];
    persist=40;
    transmit=1;
    
    figure('Color','black','Renderer','painters');
    colormap copper;
    M(paramIQ.NumFrames+3)=struct('cdata','','colormap','');
    for ifr=1:paramIQ.NumFrames
        ME=max(EF(:,:,max(1,ifr-persist):ifr),[],3);
        if isequal(ifr,1)
            imagesc([-paramIQ.Width/2 paramIQ.Width/2]*1e3,[paramIQ.DepthMin paramIQ.DepthMax]*1e3,20*log10(squeeze(ME)));
            ax=gca;
            saveas(gcf,string(ifr),'png')
        else
            imagesc(ax,[-paramIQ.Width/2 paramIQ.Width/2]*1e3,[paramIQ.DepthMin paramIQ.DepthMax]*1e3,20*log10(squeeze(ME)));
            saveas(gcf,string(ifr),'png')
        end
        set(gca,'FontSize',12,'XColor','White','YColor','White');
        colorbar('Color','white')
        axis equal
        axis tight;
        caxis(DR);
        M(ifr)=getframe(gcf);
    end
    colormap copper
    ME=max(EF(:,:,:),[],3);
    imagesc(ax,[-paramIQ.Width/2 paramIQ.Width/2]*1e3,[paramIQ.DepthMin paramIQ.DepthMax]*1e3,20*log10(squeeze(ME)));
    set(gca,'FontSize',12,'XColor','White','YColor','White');
    colorbar('Color','white')
    axis equal
    axis tight;
    caxis(DR);
    M(paramIQ.NumFrames+1:paramIQ.NumFrames+3)=getframe(gcf);
    avifilename=strrep(file_name,'.bmode.iq.mat','.ESVD_200800sepia.mp4');
    vidObj=VideoWriter(avifilename,'MPEG-4');
    vidObj.FrameRate=50;
    vidObj.Quality=100;
    open(vidObj);
    for i=1:paramIQ.NumFrames+1
        writeVideo(vidObj,M(i));
        if i == paramIQ.NumFrames
            saveas(gcf,"MIP.png")
        else
            continue
        end
    end
    clear M;
    close(vidObj);
    clear vidObj;
end






% Jamie's velocity variance analysis algorithm. Only currently works one
% frame at a time 


% We have already run SVD Example ideally twice, at least once 

% clearing variables that if they exist, will screw up the program
tic
close all



if exist('h','var')
 clear h 
end
if exist('gummy','var')
 clear gummy
end
if  exist('rightneighbour_indices','var') || exist('leftneighbour_indices','var') || exist('upneighbour_indices','var') || exist('downneighbour_indices','var') || exist('leftbottom_indices','var') || exist('rightbottom_indices','var') || exist('righttop_indices','var') || exist('lefttop_indices','var')
    clear rightneighbour_indices
    clear leftneighbour_indices
    clear righttop_indices
    clear lefttop_indices
    clear upneighbour_indices
    clear downneighbour_indices
    clear rightbottom_indices
    clear leftbottom_indices
end

% If svd_example.m has not been run, this will filter the image 
if ~exist('DD','var')
    IR=reshape(I,paramIQ.NumSamples*paramIQ.NumLines,paramIQ.NumFrames);
    QR=reshape(Q,paramIQ.NumSamples*paramIQ.NumLines,paramIQ.NumFrames);
    [UI,SI,VI]=svd(IR,'econ');
    [UQ,SQ,VQ]=svd(QR,'econ');
    ThresholdL=200;
    ThresholdH=800;
    SIF=zeros(size(SI));
    SQF=zeros(size(SQ));
    SIF(ThresholdL:ThresholdH,ThresholdL:ThresholdH)=SI(ThresholdL:ThresholdH,ThresholdL:ThresholdH);
    SQF(ThresholdL:ThresholdH,ThresholdL:ThresholdH)=SQ(ThresholdL:ThresholdH,ThresholdL:ThresholdH);
    
    IRF=UI*SIF*VI';
    QRF=UQ*SQF*VQ';
    IF=reshape(IRF,paramIQ.NumSamples,paramIQ.NumLines,paramIQ.NumFrames);
    QF=reshape(QRF,paramIQ.NumSamples,paramIQ.NumLines,paramIQ.NumFrames);
    EF=(IF.^2+QF.^2).^0.5;

    DR=[20 60];
    persist=40;
    transmit=1;
    
    for ifr=1:paramIQ.NumFrames
        ME=max(EF(:,:,max(1,ifr-persist):ifr),[],3); 
        DD(:,:,ifr) = 20*log10(squeeze(ME));
    end
end 

save DD
