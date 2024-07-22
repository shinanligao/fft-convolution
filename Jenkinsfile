@Library(['holoutils@master', 'bam@main']) _


def branch = env.CHANGE_BRANCH ? env.CHANGE_BRANCH : env.BRANCH_NAME
def editedBranch = branch.replaceAll("/", "-")

pipeline {
    agent none
    options {
        timeout(time: 2, unit: 'HOURS')
        buildDiscarder(logRotator(daysToKeepStr: '7', artifactDaysToKeepStr: '7'))
    }
    stages {
        stage('Format check') {
            agent {
                label 'macos && clang14 && rust'
            }
            environment {
                PATH = "$PATH:/usr/local/bin"
                RUST_HOME = tool name: 'rust'
                RUSTUP_HOME = "${RUST_HOME}/.."
                CARGO_HOME = "${RUST_HOME}/.."
                CI = 'true'
            }
            steps {
                script {
                    sshagent(credentials: ['ssh_key']) {
                        sh "${RUST_HOME}/cargo fmt --check"
                    }
                }
            }
        }
        stage('Build and Test'){
            parallel{
                stage('OSX Arm64') {
                    agent{
                        label "macos-arm && clang14 && rust"
                    }
                    environment{
                        PATH = "$PATH:/usr/local/bin"
                        RUST_HOME = tool name: 'rust'
                        RUSTUP_HOME = "${RUST_HOME}/.."
                        CARGO_HOME = "${RUST_HOME}/.."
                        CI = 'true'
                    }
                    stages{
                        stage('Build') {
                            steps {
                                script{
                                    sshagent (credentials: ['ssh_key']) {
                                        sh "${RUST_HOME}/cargo build --release"
                                    }
                                }
                            }
                        }
                        stage('Unit tests') {
                            steps {
                                script{
                                    sshagent (credentials: ['ssh_key']) {
                                        sh "${RUST_HOME}/cargo test --release"
                                    }
                                }
                            }
                        }
                    }
                    post{
                        cleanup{
                            cleanWs()
                        }
                    }
                }
                stage('OSX Intel') {
                    agent{
                        label "${env.BRANCH_NAME}" =~ /^release/ || "${env.BRANCH_NAME}" == 'develop' ? 'mb1' : 'macos && clang14 && rust'
                    }
                    environment{
                        PATH = "$PATH:/usr/local/bin"
                        RUST_HOME = tool name: 'rust'
                        RUSTUP_HOME = "${RUST_HOME}/.."
                        CARGO_HOME = "${RUST_HOME}/.."
                        CI = 'true'
                    }
                    stages{
                        stage('Build') {
                            steps {
                                script{
                                    sshagent (credentials: ['ssh_key']) {
                                        sh "${RUST_HOME}/cargo build --release"
                                    }
                                }
                            }
                        }
                        stage('Unit tests') {
                            steps {
                                script{
                                    sshagent (credentials: ['ssh_key']) {
                                        sh "${RUST_HOME}/cargo test --release"
                                    }
                                }
                            }
                        }
                    }
                    post{
                        cleanup{
                            cleanWs()
                        }
                    }
                }
            }
        }
    }
}
