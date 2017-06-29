'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Demo for the evaluation of video summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Package to evaluate and plot summarization results
% on the SumMe dataset
%
%%%%%%%%
% publication: Gygli et al. - Creating Summaries from User Videos, ECCV 2014
% author:      Michael Gygli, PhD student, ETH Zurich,
% mail:        gygli@vision.ee.ethz.ch
% date:        05-16-2014
'''
import scipy.io
import warnings
import numpy as np
import matplotlib.pyplot as plt



def evaluateSummary(summary_selection,videoName,HOMEDATA):
     '''Evaluates a summary for video videoName (where HOMEDATA points to the ground truth file)
     f_measure is the mean pairwise f-measure used in Gygli et al. ECCV 2013
     NOTE: This is only a minimal version of the matlab script'''
     # Load GT file

     # print (videoName)

     # print (len(summary_selection))
     # print ('=====')


     gt_file=HOMEDATA+'/'+videoName+'.mat'
     gt_data = scipy.io.loadmat(gt_file)

     user_score=gt_data.get('user_score')
     nFrames=user_score.shape[0];
     nbOfUsers=user_score.shape[1];

     # summary_selection = user_score[:,0]

     # print('{} == {}'.format(user_score.shape,summary_selection.shape))

     # Check inputs
     # if len(summary_selection) < nFrames:
     #      warnings.warn('Pad selection with %d zeros!' % (nFrames-len(summary_selection)))
     #      summary_selection = np.array(summary_selection)
     #      summary_selection = np.append(summary_selection,np.zeros(nFrames-len(summary_selection)))
     #      # summary_selection.extend(np.zeros(nFrames-len(summary_selection)))
     #
     # elif len(summary_selection) > nFrames:
     #      warnings.warn('Crop selection (%d frames) to GT length' %(len(summary_selection)-nFrames))
     #      summary_selection=summary_selection[0:nFrames];


     # Compute pairwise f-measure, summary length and recall

     summary_indicator=np.array(map(lambda x: (1 if x>0 else 0),summary_selection));
     # print (len(summary_indicator))
     # print('{}******'.format(float(np.sum(summary_indicator)) / float(len(summary_indicator))))
     # summary_indicator = summary_selection
     user_intersection=np.zeros((nbOfUsers,1));
     user_union=np.zeros((nbOfUsers,1));
     user_length=np.zeros((nbOfUsers,1));
     for userIdx in range(0,nbOfUsers):
         gt_indicator=np.array(map(lambda x: (1 if x>0 else 0),user_score[:,userIdx]))
         gt_indicator = sample(gt_indicator,nFrames,gt_data.get('FPS'))
         gt_indicator = gt_indicator[:len(summary_indicator)]
         # print (len(gt_indicator))
         # print('{}====='.format(float(np.sum(gt_indicator))/float(len(gt_indicator))))
         user_intersection[userIdx]=np.sum(gt_indicator*summary_indicator);
         user_union[userIdx]=sum(np.array(map(lambda x: (1 if x>0 else 0),gt_indicator + summary_indicator)));

         user_length[userIdx]=sum(gt_indicator)

     recall=user_intersection/user_length;
     p=user_intersection/np.sum(summary_indicator);

     f_measure=[]
     for idx in range(0,len(p)):
          if p[idx]>0 or recall[idx]>0:
               f_measure.append(2*recall[idx]*p[idx]/(recall[idx]+p[idx]))
          else:
               f_measure.append(0)
     print ('{}  ----'.format(f_measure[0]))
     f_measure=np.mean(f_measure)

     return (f_measure*100)


def sample(gt,num_frames,fps):
    # for i in range(0, num_frames - 1, fps):
    #     file.write('{} {}'.format(trainY[0][i], trainY[0][min(i + int(fps / 2), num_frames - 1)]))
    #     if i + fps < num_frames - 1:
    #         file.write(' ')
    return [gt[i] for i in range(0, num_frames - 1, fps/2)]