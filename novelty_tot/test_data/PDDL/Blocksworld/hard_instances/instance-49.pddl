

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(ontable c)
(on d b)
(ontable e)
(on f a)
(clear c)
(clear f)
)
(:goal
(and
(on b e)
(on c d)
(on d a)
(on f b))
)
)


