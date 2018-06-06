package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

/**
 * 实现了基于加权多样性和精度的集成分类算法。每个基分类器的权重是根据多样性和精度的加权来确定的
 * @author 张本才
 *
 */


public class DiversityAccuracyEnsemble extends AbstractClassifier implements MultiClassClassifier{
	private static final long serialVersionUID = 1L;

	/**
	 * Type of classifier to use as a component classifier.
	 */
	public ClassOption learnerOption = new ClassOption("learner", 'l', "Classifier to train.", Classifier.class, 
			"trees.HoeffdingTree -e 2000000 -g 100 -c 0.01");

	/**
	 * Number of component classifiers.
	 */
	public IntOption memberCountOption = new IntOption("memberCount", 'n',
			"The maximum number of classifiers in an ensemble.", 10, 1, Integer.MAX_VALUE);

	/**
	 * Chunk size.
	 */
	public IntOption chunkSizeOption = new IntOption("chunkSize", 'c',
			"The chunk size used for classifier creation and evaluation.", 500, 1, Integer.MAX_VALUE);

	/**
	 * Determines the maximum size of model (evaluated after every chunk).
	 */
	public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm', "Maximum memory consumed by ensemble.",
			33554432, 0, Integer.MAX_VALUE);
	
	/**
	 * The weight for accuracy when weighting the accuracy and diversity as the weight of base classifier
	 * in ensemble
	 * The default value of alpha  is 0.5
	 */
	
	public FloatOption alphaOption = new FloatOption("alpha", 'a', "The weight for accuracy when weighting the accuracy\\r\\n"
			+ " and diversity as the weight of base classifier in ensemble", 0.5, 0, 1) ;

	/**
	 * The weights of stored classifiers. 
	 * weights[x][0] = weight
	 * weights[x][1] = classifier number in learners
	 */
	protected double[][] weights;
	
	/**
	 * Class distributions.
	 */
	protected long[] classDistributions;
	
	/**
	 * Ensemble classifiers.
	 */
	protected Classifier[] learners;
	
	/**
	 * Number of processed examples.
	 */
	protected int processedInstances;
	
	/**
	 * Candidate classifier.
	 */
	protected Classifier candidate;
	
	/**
	 * Current chunk of instances.
	 */
	protected Instances currentChunk;
	
	

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();

		super.prepareForUseImpl(monitor, repository);
	}
	

	@Override
	public void resetLearningImpl() {
		this.currentChunk = null;
		this.classDistributions = null;
		this.processedInstances = 0;
		this.learners = new Classifier[0];

		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		this.initVariables();

		this.classDistributions[(int) inst.classValue()]++;  //保存类分布
		this.currentChunk.add(inst);
		this.processedInstances++;

		if (this.processedInstances % this.chunkSizeOption.getValue() == 0) { //如果收集的已初始化的实例够一个数据块
			this.processChunk();
		}
	}

	/**
	 * Determines whether the classifier is randomizable.
	 */
	public boolean isRandomizable() {
		return false;
	}

	/**
	 * Predicts a class for an example.
	 */
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();

		if (this.trainingWeightSeenByModel > 0.0) {
			for (int i = 0; i < this.learners.length; i++) {
				if (this.weights[i][0] > 0.0) {
					DoubleVector vote = new DoubleVector(this.learners[(int) this.weights[i][1]].getVotesForInstance(inst));

					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						// scale weight and prevent overflow
						vote.scaleValues(this.weights[i][0] / (1.0 * this.learners.length + 1.0));
						combinedVote.addValues(vote);
					}
				}
			}
		}
		
		//combinedVote.normalize();
		return combinedVote.getArrayRef();
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	@Override
	public Classifier[] getSubClassifiers() {
		return this.learners.clone();
	}
	
	//使用Q统计量计算每对分类器之间的差异性
	private double computeDiversityPerPair(Classifier c1, Instances chunk, Classifier c2) {

		int num00, num01, num10, num11 ;
		double qValue ;
		num00 = 0;
		num01 = 0; 
		num10 = 0;
		num11 = 0;
		for(int j=0; j<chunk.numInstances(); j++) {
			//同时分错
			if( (!c1.correctlyClassifies(chunk.get(j))) && (!c2.correctlyClassifies(chunk.get(j))) ) {
				num00 ++ ;
			}
			//其中一个分错，另一个分对
			else if(!c1.correctlyClassifies(chunk.get(j)) && c2.correctlyClassifies(chunk.get(j))) {
				num01 ++ ;
			}
			else if(c1.correctlyClassifies(chunk.get(j)) && !c2.correctlyClassifies(chunk.get(j))) {
				num10 ++ ;
			}
			//同时分对
			else if(c1.correctlyClassifies(chunk.get(j)) && c2.correctlyClassifies(chunk.get(j))) {
				num11 ++ ;
			}
		}
		qValue = (num11*num00 - num10*num01)/(1.0*(num11*num00 + num10*num01)) ;
		
//		return (1.0 - qAve/(1.0*numLearners)) ;
		//归一化
		return 0.5*(1-qValue) ;
	}
	
	private double computeDiversityForNewClassifier(Classifier[] ensemble, Classifier newClassifier
			, Instances chunk) {
		int numLearners = ensemble.length ;
		double sumQ = 0 ;
		for(int i=0; i<numLearners; i++) {
			double tmpQ = computeDiversityPerPair(newClassifier, chunk, ensemble[i]) ;
			sumQ += tmpQ ;
		}
		return sumQ/(1.0*numLearners) ; 
	}
	
	/**
	 * 使用Q统计量计算ensemble中的一个分类器与其他分类器的差异
	 * @param ensemble
	 * @param chunk
	 * @return
	 */
	private double[] computeDiversity(Classifier[] ensemble, Instances chunk) {
		int numLearners = ensemble.length ;
		double[] div = new double[numLearners]  ;
		double divAvg ;
		double divTmp ;
		for(int i=0; i<numLearners; i++) {
			divAvg = 0 ;
			for(int j=0; j<numLearners; j++) {
				if(j != i) {
					//两两计算Q值并相加
					divTmp = computeDiversityPerPair(ensemble[i], chunk, ensemble[j]) ;
					divAvg += divTmp ;
				}
			}
			//对两两
			div[i] = divAvg/(1.0*numLearners) ;
		}
		return div ;
	}

	/**
	 * Processes a chunk of instances.
	 * This method is called after collecting a chunk of examples.
	 */
	protected void processChunk() {
		Classifier addedClassifier = null;
		int numLearners = this.learners.length ;
		double mse_r = this.computeMseR();
		double[] divEnsemble = new double[numLearners] ;
		double[] accuracy = new double[numLearners];
		double candidateAcc = 0 ;
		double candidateDiv = 0 ;
		double alpha = alphaOption.getValue() ;

		// Compute weights
		candidateAcc = 1.0 / (mse_r + Double.MIN_VALUE);
		if(alpha != 1.0) {
			candidateDiv = computeDiversityForNewClassifier(this.learners, this.candidate, currentChunk) ;
		}
		double candidateClassifierWeight = alpha*candidateAcc + (1-alpha)*candidateDiv ;

		for (int i = 0; i < this.learners.length; i++) {
			accuracy[i] = 1.0 / (mse_r + this.computeMse(this.learners[(int) this.weights[i][1]],
					this.currentChunk) + Double.MIN_VALUE);
			
		}
		//计算多样性
		if(alpha != 1.0) {
			divEnsemble = computeDiversity(this.learners, currentChunk);
		}
		
		for(int i=0; i < this.learners.length; i++) {
			weights[i][0] = (1-alpha)*divEnsemble[i] + alpha*accuracy[i] ;
		}
		if (this.learners.length < this.memberCountOption.getValue()) {
			// Train and add classifier
			addedClassifier = this.addToStored(this.candidate, candidateClassifierWeight);
		} else {
			// Substitute poorest classifier
			int poorestClassifier = this.getPoorestClassifierIndex();

			if (this.weights[poorestClassifier][0] < candidateClassifierWeight) {
				this.weights[poorestClassifier][0] = candidateClassifierWeight;
				addedClassifier = this.candidate.copy();
				this.learners[(int) this.weights[poorestClassifier][1]] = addedClassifier;
			}
		}

		// train classifiers
		for (int i = 0; i < this.learners.length; i++) {
			this.trainOnChunk(this.learners[(int) this.weights[i][1]]);
		}

		this.classDistributions = null;
		this.currentChunk = null;
		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();

		this.enforceMemoryLimit();
	}

	/**
	 * Checks if the memory limit is exceeded and if so prunes the classifiers in the ensemble.
	 */
	protected void enforceMemoryLimit() {
		double memoryLimit = this.maxByteSizeOption.getValue() / (double) (this.learners.length + 1);

		for (int i = 0; i < this.learners.length; i++) {
			((HoeffdingTree) this.learners[(int) this.weights[i][1]]).maxByteSizeOption.setValue((int) Math
					.round(memoryLimit));
			((HoeffdingTree) this.learners[(int) this.weights[i][1]]).enforceTrackerLimit();
		}
	}

	/**
	 * Computes the MSEr threshold.
	 * 
	 * @return The MSEr threshold.
	 */
	protected double computeMseR() {
		double p_c;
		double mse_r = 0;

		for (int i = 0; i < this.classDistributions.length; i++) {
			p_c = (double) this.classDistributions[i] / (double) this.chunkSizeOption.getValue();
			mse_r += p_c * ((1 - p_c) * (1 - p_c));
		}

		return mse_r;
	}
	
	/**
	 * Computes the MSE of a learner for a given chunk of examples.
	 * @param learner classifier to compute error
	 * @param chunk chunk of examples
	 * @return the computed error.
	 */
	protected double computeMse(Classifier learner, Instances chunk) {
		double mse_i = 0;

		double f_ci;
		double voteSum;

		for (int i = 0; i < chunk.numInstances(); i++) {
			try {
				voteSum = 0;
				for (double element : learner.getVotesForInstance(chunk.instance(i))) {
					voteSum += element;
				}

				if (voteSum > 0) {
					f_ci = learner.getVotesForInstance(chunk.instance(i))[(int) chunk.instance(i).classValue()]
							/ voteSum;
					mse_i += (1 - f_ci) * (1 - f_ci);
				} else {
					mse_i += 1;
				}
			} catch (Exception e) {
				mse_i += 1;
			}
		}

		mse_i /= this.chunkSizeOption.getValue();

		return mse_i;
	}
	
	/**
	 * Adds ensemble weights to the measurements.
	 */
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measurements = new Measurement[(int) this.memberCountOption.getValue()];

		for (int m = 0; m < this.memberCountOption.getValue(); m++) {
			measurements[m] = new Measurement("Member weight " + (m + 1), -1);
		}

		if (this.weights != null) {
			for (int i = 0; i < this.weights.length; i++) {
				measurements[i] = new Measurement("Member weight " + (i + 1), this.weights[i][0]);
			}
		}

		return measurements;
	}

	/**
	 * Adds a classifier to the storage.
	 * 
	 * @param newClassifier
	 *            The classifier to add.
	 * @param newClassifiersWeight
	 *            The new classifiers weight.
	 */
	protected Classifier addToStored(Classifier newClassifier, double newClassifiersWeight) {
		Classifier addedClassifier = null;
		Classifier[] newStored = new Classifier[this.learners.length + 1];
		double[][] newStoredWeights = new double[newStored.length][2];

		for (int i = 0; i < newStored.length; i++) {
			if (i < this.learners.length) {
				newStored[i] = this.learners[i];
				newStoredWeights[i][0] = this.weights[i][0];
				newStoredWeights[i][1] = this.weights[i][1];
			} else {
				newStored[i] = addedClassifier = newClassifier.copy();
				newStoredWeights[i][0] = newClassifiersWeight;
				newStoredWeights[i][1] = i;
			}
		}
		this.learners = newStored;
		this.weights = newStoredWeights;

		return addedClassifier;
	}
	
	/**
	 * Finds the index of the classifier with the smallest weight.
	 * @return
	 */
	private int getPoorestClassifierIndex() {
		int minIndex = 0;
		
		for (int i = 1; i < this.weights.length; i++) {
			if(this.weights[i][0] < this.weights[minIndex][0]){
				minIndex = i;
			}
		}
		
		return minIndex;
	}
	
	/**
	 * Initiates the current chunk and class distribution variables.
	 */
	private void initVariables() {
		if (this.currentChunk == null) {
			this.currentChunk = new Instances(this.getModelContext());
		}

		if (this.classDistributions == null) {
			this.classDistributions = new long[this.getModelContext().classAttribute().numValues()];

			for (int i = 0; i < this.classDistributions.length; i++) {
				this.classDistributions[i] = 0;
			}
		}
	}
	
	/**
	 * Trains a component classifier on the most recent chunk of data.
	 * 
	 * @param classifierToTrain
	 *            Classifier being trained.
	 */
	private void trainOnChunk(Classifier classifierToTrain) {
		for (int num = 0; num < this.chunkSizeOption.getValue(); num++) {
			classifierToTrain.trainOnInstance(this.currentChunk.instance(num));
		}
	}

}
