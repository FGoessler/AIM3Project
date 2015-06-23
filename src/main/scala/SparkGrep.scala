package spark.example

/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object SparkGrep {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkGrep").setMaster("local[*]")
    val sc = new SparkContext(conf)

    /* read in plot descriptions */
    // TODO

    /* calculate TF-IDF vectors */
    // TODO


    /* load movie to genre mapping */
    val inputFile = sc.textFile("genres.list", 2).cache().sample(withReplacement = false, 0.1, 5)
    val moviesWithGenres = inputFile.map(line => {
      val s = line.split("\t+")
      (s(0), s(1))
    })

    // TODO: replace with actual calculated TF-IDF vectors
    val moviesWithTFIDFVectors = moviesWithGenres.map(m => m._1).distinct().map(m => (m, Vectors.dense(Array(1.0, 2.0, 3.0, 4.0))))

    /* join TF-IDF vectors with genres to get labeled data */
    val moviesWithGenresAndTFIDFVector = moviesWithGenres.join(moviesWithTFIDFVectors)
     /* split data into training (70%) and test (30%) */
    val splits = moviesWithGenresAndTFIDFVector.randomSplit(Array(0.7, 0.3), seed = 11L)
    val moviesWithGenresAndTFIDFVector_training = splits(0).cache()
    val moviesWithGenresAndTFIDFVector_test = splits(1).cache()


    /* train SVMs for genres */
    val comedyModel = trainModelForGenre(moviesWithGenresAndTFIDFVector_training, "Comedy")
    // TODO: train more SVMs for other genres

    /* use trained models to predict genres of the test data */
    moviesWithGenresAndTFIDFVector_test
      .map(m => (m._1, m._2._1, comedyModel.predict(m._2._2)))
      // TODO: make actual precision analysis instead of just printing simple text
      .foreach(m => println("%s is a %s was predicted with %f as 'Comedy'".format(m._1, m._2, m._3)))

    System.exit(0)
  }

  def trainModelForGenre(trainingData: RDD[(String, (String, Vector))], genre: String): SVMModel = {
    val trainigDataForComedy = trainingData.map(m => LabeledPoint(if (m._2._1 == genre) 1.0 else 0.0 , m._2._2))
    val model = SVMWithSGD.train(trainigDataForComedy, 100)
    model.clearThreshold()
    model
  }
}