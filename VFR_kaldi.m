function feature=VFR_kaldi(input,fsamp,winshft,VAD)
% Function performs VFR analysis based on approximated frame entropy,
% according to (You et al, 2004).
% 'input' is the raw speech signal
% 'fsamp' is the sampling rate of 'input'
% 'winshift' is the window shift in ms during the intial oversampling (e.g. 2.5 ms)
% 'VAD' is voice activity detection results for the input speech-
% it consists of a vector of binary values where '1' refers to speech, and '0' refers to nonspeech

doDisp=0; % Display option- '1' outputs a graphic display of VFR analysis, '0' does not

% -------------------------------------------------------------------------
winlen      = 25;             % window length in 100 nsec
nfft        = 512;            % fft size
cepnum      = 19;             % number of cepstral coefficients
liftercoe   = 22;             % liftering coefficient (e.g. 22)
numchan     = 23;             % number of channels of the MEL filter bank (e.g. 26)
% -------------------------------------------------------------------------

winlen=round(winlen*10^(-3)*fsamp);
winshft=winshft*10^(-3)*fsamp;
FrameNo=round(length(input)/winshft);

% initialize MEL filter bank
fbank = initfiltb(winlen, numchan, fsamp, nfft);

% initialize lifter coefficients
lifter = (1 + (liftercoe/2)*sin((pi/liftercoe)*(0:cepnum)) );

% change signal (a vector) into frame (a matrix), where each collum is a frame
frmwin = sig2fm(input, winlen, winshft, FrameNo);
[winlen, framenum]=size(frmwin);

% Hamming window each frame
frmwin = frmwin .* (hamming(winlen) * ones(1, framenum));

% FFT
ffto=abs(fft(frmwin,nfft));

% MEL filtering
fb=fbank*ffto(1 : (nfft/2), :);

% take logarithm of MEL filter output
fbfloor=mean(mean(fb))*0.00001;
logfb=log(max(fb, fbfloor*rand(size(fb))));

% take DCT
mfcco=dct(logfb);
mfcco=mfcco(1:cepnum+1,:);

% lifer MFCCs
mfcco=mfcco.*(lifter'*ones(1,framenum));

% determine frame entropies
ENT_TEMP=find_entropy([input;input(end-winlen:end)],fsamp,numchan,nfft);

keep=ones(1,framenum);
ENT=[];
ctr=1;
rate=8;
ho=1;
for fr=7:6:size(fb,2)-6
    curr_VAD=VAD(:,fr-3:fr+2);
    if(sum(curr_VAD)==0)
        start=fr-3-ho-1;
        finish=fr+2;
        keep(start:rate:finish)=0;
        ho=mod(finish-start,rate);
    else
        ho=0;
        ENT=[ENT ENT_TEMP(ctr)];
    end
    ctr=ctr+1;
end

if(~isempty(ENT))
    Mx=max(ENT);
    Md=median(ENT);
    Mn=min(ENT);
else
    Mx=1;
    Md=1;
    Mn=1;
end

% perform VFR analysis, according to Section 3 of (You et al., 2004)
ctr=1;
ent_ctr=1;
w1=0.6; %0.7
w2=0.7; %0.8
w3=0.4; %0.5
T1=w1*Mx+(1-w1)*Md;
T2=(1-w2)*Mx+w2*Md;
T3=(1-w3)*Md+w3*Mn;
ho=1;
for fr=7:6:size(fb,2)-6
    curr_VAD=VAD(:,fr-3:fr+2);
    if(sum(curr_VAD)>0)
        curr_frames=mfcco(:,fr-3:fr+2);
        curr_ent=ENT(ent_ctr);
        ent_ctr=ent_ctr+1;
        if(curr_ent>T1)
            rate=2; %2
        elseif(curr_ent>T2)
            rate=3; %3
        elseif(curr_ent>T3)
            rate=4; %4
        else
            rate=5; %5
        end
        start=fr-3-ho-1;
        finish=fr+2;
        keep(start:rate:finish)=0;
        ho=mod(finish-start,rate);
        plot_ent(ent_ctr-1)=curr_ent;
    else
        ho=0;
        rate=8; %8
        curr_ent=0;
    end
    plot_rate(ctr)=rate;
    ctr=ctr+1;
end
% S1=size(mfcco,2);
% mfcco=mfcco(:,keep==0);


% mfcc1 = mfcco - mean(mfcco, 2) * ones(1, size(mfcco, 2));

% determine deltas and double-deltas, and concatenate into a matrix of
% feature vectors
% dt1=deltacc(mfcc1);
% dt2=deltacc(dt1);
% mfcc1=[mfcc1;dt1;dt2];
% % mfcc1=mfcc1(:,keep==0);

% if(doDisp)
%     figure;
%     subplot(3,1,1);plot(input);axis tight;
%     title('Speech Signal');
%     subplot(3,1,2);plot(plot_ent);axis tight;
%     title('Frame Entropy Values');
%     subplot(3,1,3);imagesc(ones(10,1)*keep);colormap(gray);
%     title('Selected Frames');
% end
if(doDisp)
    figure;
    subplot(4,1,1);plot(input);axis tight;
    title('Speech Signal');
    subplot(4,1,2);plot(plot_ent);axis tight;
    title('Frame Entropy Values');
    subplot(4,1,3);plot(plot_rate);axis tight;
    title('Frame Entropy Values');
    subplot(4,1,4);imagesc(ones(10,1)*keep);colormap(gray);
    title('Selected Frames');
end
% perform cepstral mean subtraction (CMS)
% mfcco=mfcco'-mean(mfcco');
% mfcco = mfcco - mean(mfcco, 2) * ones(1, size(mfcco, 2));

% determine deltas and double-deltas, and concatenate into a matrix of
% feature vectors
% dt1=deltacc(mfcco);
% dt2=deltacc(dt1);
% mfcco=[mfcco;dt1;dt2];
% mfcco=mfcco(:,keep==0);
feature = 1-keep;


% ---------------------------------------------------------------
% ---------------------------------------------------------------
function ENT=find_entropy(speech,fsamp,numchan,nfft)
% Function approximates the feature space entropy of an input speech signal
% on a frame-by-frame basis, according to Eq. 5 of (You et al., 2004).
% 'speech' is the raw speech signal
% 'fsamp' is the sampling rate of 'speech'
% 'numchan' is the number of Mel-channels used during short-time analysis
% 'nfft' is the size of the DFT used during short-time analysis
% The function returns 'ENT', a vector of frame entropies

winlen=round(25*10^(-3)*fsamp);
winshft=2.5*10^(-3)*fsamp;
FrameNo = ceil((length(speech) - winlen) / winshft);

% initialize MEL filter bank
fbank=initfiltb(winlen,numchan,fsamp,nfft);

% change signal (a vector) into frame (a matrix), where each collum is a frame
frmwin=sig2fm(speech,winlen,winshft,FrameNo);
[winlen,framenum]=size(frmwin);

% Hamming window each frame
frmwin=frmwin.*(hamming(winlen)*ones(1,framenum));

% FFT
ffto=abs(fft(frmwin,nfft));

% MEL filtering
fb=fbank*ffto(1:(nfft/2),:);

% Approximate entropy of each MFCC frame according to
% Eq. 5 of (You et al., 2004)
ctr=1;
for fr=7:6:framenum-6
    TR=0;
    curr_frames=fb(:,fr-6:fr+5);
    MU=mean(curr_frames')';
    MU=MU*ones(1,size(curr_frames,2));
    SIGMA=(curr_frames-MU)*(curr_frames-MU)';
    for k=1:numchan
        TR=TR+SIGMA(k,k);
    end
    TR=log(TR)/log(exp(1));
    ENT(ctr)=numchan*log((2*pi)^0.5)/log(exp(1))+TR;
    ctr=ctr+1;
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
function mels=mel(freq)
% change frequency from Hz to mel
mels=1127.0*log(1.0+(freq/700.0));

% ---------------------------------------------------------------
% ---------------------------------------------------------------
function wins=sig2fm(input,winlen,winshft,frameno)
% put vector into matrix, each column is a frame.
% The rest of signal that is less than one frame is discarded
% winlen, winshft are in number of sample, notice winshft is not limited to
% integer
input=input(:);
wins=zeros(winlen, frameno);
for i=1:frameno
    b=round((i-1)*winshft);
    c=min(winlen,length(input)-b);
    wins(1:c,i)=input(b+1:min(length(input),b+winlen));
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
function fbank=initfiltb(framelen,numchan,fsamp,nfft)
% triangle shape melfilter initialization

fftfreqs=((0:(nfft/2-1))/nfft)*fsamp;  % frequency of each fft point (1-fsamp/2)
melfft=mel(fftfreqs);   % mel of each fft point
mel0=0;
mel1=mel(fsamp/2);       % highest mel
% mel0=mel(20);
% mel1=mel(3700);       % highest mel
melmid=((1:numchan)/(numchan+1))*(mel1-mel0)+mel0; % middle mel of each filter
fbank=zeros(numchan,nfft/2);

% non overlaping triangle window is used to form the mel filter
for k=2:(nfft/2)  % for each fft point, to all the filters,do this:
    chan=max([0 find(melfft(k)>melmid)]); % the highest index of melfft that is larger than the middle mel of all channels
    if(chan==0)  % only the first filter cover here
        fbank(1,k)=(melfft(k)-mel0)/(melmid(1)-mel0);
    elseif(chan==numchan)  % only the last filter covered here
        fbank(numchan,k)=(mel1-melfft(k))/(mel1-melmid(chan));
    else                   % for any other part, there will be two filter cover that frequency, in the complementary manner
        fbank(chan,k)=(melmid(chan+1)-melfft(k))/(melmid(chan+1)-melmid(chan));
        fbank(chan+1,k)=1-fbank(chan,k);  % complementary
    end
end

% ---------------------------------------------------------------
% ---------------------------------------------------------------
function dt=deltacc(input)
% calculates derivatives of a matrix, whose columns are feature vectors

tmp=0;
for cnt=1:2
    tmp=tmp+cnt*cnt;
end
nrm=1/(2*tmp);
dt=zeros(size(input));
rows=size(input,1);
cols=size(input,2);
for col=1:cols
    for cnt=1:2
        inx1=col-cnt;
        inx2=col+cnt;
        if(inx1<1)
            inx1 = 1;
        end
        if(inx2>cols)
            inx2 = cols;
        end
        dt(:,col)=dt(:,col)+(input(:,inx2)-input(:,inx1))*cnt;
    end
end
dt=dt*nrm;
% ---------------------------------------------------------------



