package com.github.spark.kmeans

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg.Vectors

object KMeans {

  def main(args: Array[String]) {

    val conf = new SparkConf()
    val sc = new SparkContext(conf)

    val data = sc.textFile(input) //Loads data

    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    //new k-means model
    val initMode = "k-means||"
    val numClusters = 2  //num
    val numIterations = 100
    val model = new KMeans().
      setInitializationMode(initMode).
      setK(numClusters).
      setMaxIterations(numIterations).
      run(parsedData)

    //print the result
    println(parsedData.map(v=> v.toString() + " belong to cluster :" +model.predict(v)).collect().mkString("\n"))

    val centers = model.clusterCenters
    println("centers")

    for (i <- 0 to centers.length - 1) {
      println(centers(i)(0) + "\t" + centers(i)(1))
    }

    //computing the squared errors
    val WSSSE = model.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

  }
}