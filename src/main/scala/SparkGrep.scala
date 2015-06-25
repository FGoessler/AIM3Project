package spark.example

/* SimpleApp.scala */

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SparkGrep {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkGrep").setMaster("local[*]")
    val sc = new SparkContext(conf)

    /* ####################### PREPROCESSING ####################### */

    /* predefined list of all genres that we analyze */
    val genreList = sc.broadcast(Array("Comedy", "Drama")) // TODO: add more genres

    /* read in plot descriptions */
    // TODO

    /* calculate TF-IDF vectors */
    // TODO


    /* load movie to genre mapping */
    val inputFile = sc.textFile("genres.list", 2).cache().sample(withReplacement = false, 0.1, 5)   //TODO: remove sampling
    val moviesWithGenres = inputFile.map(line => {
      val s = line.split("\t+")
      val genreVector = Vectors.dense(genreList.value.map(f => if (f == s(1)) 1.0 else 0.0))
      (s(0), genreVector)
    }).reduceByKey((v1, v2) => Vectors.dense(v1.toArray.toList.zip(v2.toArray).map(w => Math.min(1.0, w._1 + w._2)).toArray))

    /* create a list of all available movie titles */
    // TODO: maybe generate this from the plot description input cause there shouldn't be any duplicates
    val movieTitles = moviesWithGenres.map(_._1).distinct()

    // TODO: replace with actual calculated TF-IDF vectors
    val moviesWithTFIDFVectors = movieTitles.map(m => (m, Vectors.dense(Array(1.0, 2.0, 3.0, 4.0))))

    /* join TF-IDF vectors with genres to get labeled data */
    val moviesWithGenresAndTFIDFVector = moviesWithGenres.join(moviesWithTFIDFVectors)
     /* split data into training (70%) and test (30%) */
    val splits = moviesWithGenresAndTFIDFVector.randomSplit(Array(0.7, 0.3), seed = 11L)
    val moviesWithGenresAndTFIDFVector_training = splits(0).cache()
    val moviesWithGenresAndTFIDFVector_test = splits(1).cache()


    /* ####################### TRAINING ####################### */

    /* train SVMs for genres */
    var modelsLocal: Seq[SVMModel] = Seq()
    for(i <- 0 to genreList.value.length - 1) {
      modelsLocal = modelsLocal :+ trainModelForGenre(moviesWithGenresAndTFIDFVector_training, i)
    }
    val models = sc.broadcast(modelsLocal.toArray)

    /* ####################### TESTING ####################### */

    /* use trained models to predict genres of the test data */
    val res = moviesWithGenresAndTFIDFVector_test
      .map(m => (m._1, m._2._1, generatePredictedGenreVector(m._2._2, models.value)))



    /* ####################### EVALUATION ####################### */

    /* output some textual info about predicted and expected genres for the movies */
    res.foreach(m => {
      val genres = genreList.value
      val expectedGenreVector = m._2
      val predictedGenreVector = m._3

      val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray).zip(genres)
      val notPredicatedGenres = zipped.flatMap(v => if (v._1._1 > v._1._2) Seq(v._2) else Seq()).mkString(",")
      val tooMuchPredicatedGenres = zipped.flatMap(v => if (v._1._1 < v._1._2) Seq(v._2) else Seq()).mkString(",")
      val correctlyPredictedGenres = zipped.flatMap(v => if ((v._1._1 == 1) && (v._1._2 == 1)) Seq(v._2) else Seq()).mkString(",")

      println("%s: correctly classified: [%s] | not classified: [%s] | too much classified: [%s]"
        .format(m._1, correctlyPredictedGenres, notPredicatedGenres, tooMuchPredicatedGenres))
    })

    /* calculate error rate */
    val precision = res.map(m => {
      val expectedGenreVector = m._2
      val predictedGenreVector: Vector = m._3
      val zipped = expectedGenreVector.toArray.toList.zip(predictedGenreVector.toArray)

      val errors = zipped.map(v => if (v._1 != v._2) 1 else 0).sum
      (errors.toLong, zipped.size.toLong)
    }).reduce((v1, v2) => (v1._1 + v2._1, v1._1 + v2._1))

    val errorPercentage = precision._1.toDouble / precision._2.toDouble

    println("%d wrong classifications, %d right classifications -> error rate of %f"
      .format(precision._1, precision._2 - precision._1, errorPercentage))

    System.exit(0)
  }

  def trainModelForGenre(trainingData: RDD[(String, (Vector, Vector))], genreIndex: Int): SVMModel = {
    val trainingDataForGenre = trainingData.map(m => LabeledPoint(m._2._1(genreIndex), m._2._2))
    val model = SVMWithSGD.train(trainingDataForGenre, 100)
    model.clearThreshold()
    model
  }

  val predictionThreshold = 0.5
  def generatePredictedGenreVector(tfIDFVector: Vector, genreModels: Array[SVMModel]): Vector = {
    val prediction = genreModels.map(model => model.predict(tfIDFVector))
    prediction.map(f => if (f > predictionThreshold) 1.0 else 0.0)
    Vectors.dense(prediction)
  }

}