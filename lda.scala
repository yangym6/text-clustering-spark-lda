package com.github.spark.lda

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.DistributedLDAModel

object LDA {

  def main(args: Array[String]) {

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val data = sc.textFile(input) //Loads data

    val parsedData = data.map(s => Vectors.dense(s.trim.split(' ').map(_.toDouble)))
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    //Trains a LDA model
    val ldaModel = new LDA().
      setK(3).
      setDocConcentration(5).
      setTopicConcentration(5).
      setMaxIterations(20).
      setSeed(0L).
      setCheckpointInterval(10).
      setOptimizer("em").
      run(corpus)

    // Describe topics of words
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize))
      { print(" " + topics(word, topic)); }
      println()
    }
    
    ldaModel.describeTopics(2)

    // Describe topics of documents
    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    distLDAModel.topicDistributions.collect.foreach(println)

  }
}